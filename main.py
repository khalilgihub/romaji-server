"""
MULTI-MODEL REAL-TIME ROMAJI ENGINE (v8.0-SPEED)
Powered by Async DeepSeek + Groq + Parallel Dictionary Lookups
Designed for Zero-Latency Music/Lyrics Applications
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from openai import AsyncOpenAI
import os
import re
import hashlib
import json
import redis.asyncio as redis
from bs4 import BeautifulSoup
import asyncio
import time
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import logging
import urllib.parse
import aiohttp

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real-Time Romaji Engine", version="8.0.0-SPEED")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ===== CONFIGURATION =====
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
REDIS_URL = os.environ.get("REDIS_URL")

# Timeout Config (Crucial for music sync)
RESEARCH_TIMEOUT = 1.2  # Max time to wait for Jisho/RomajiDesu
LLM_TIMEOUT = 3.0       # Max time for AI response

MODELS_CONFIG = {
    "deepseek": {
        "name": "deepseek-chat",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "weight": 2.5, # DeepSeek is the accuracy king
        "enabled": bool(DEEPSEEK_API_KEY)
    },
    "llama-groq": {
        "name": "llama-3.3-70b-versatile",
        "provider": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "weight": 1.5, # Groq is the speed king
        "enabled": bool(GROQ_API_KEY)
    }
}

# Data Models
@dataclass
class WordAnalysis:
    surface: str
    reading: Optional[str]
    romaji: Optional[str]

# Globals
ai_clients = {}
redis_client = None
tagger = None
kakasi_converter = None
memory_cache = {}

# ===== SYSTEMS INITIALIZATION =====
def initialize_systems():
    global tagger, kakasi_converter, ai_clients, redis_client
    
    # 1. MeCab (Morphological Analysis)
    try:
        import fugashi
        try:
            tagger = fugashi.Tagger()
        except:
            import unidic_lite
            tagger = fugashi.Tagger(f'-d {unidic_lite.DICDIR}')
    except:
        logger.error("MeCab initialization failed")

    # 2. PyKakasi (Baseline Converter)
    try:
        import pykakasi
        k = pykakasi.kakasi()
        k.setMode("H", "a") # Hiragana to ascii
        k.setMode("K", "a") # Katakana to ascii
        k.setMode("J", "a") # Japanese to ascii
        k.setMode("r", "Hepburn")
        kakasi_converter = k.getConverter()
    except:
        logger.error("Kakasi initialization failed")
        
    # 3. AI Clients
    if MODELS_CONFIG["deepseek"]["enabled"]:
        ai_clients["deepseek"] = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=MODELS_CONFIG["deepseek"]["base_url"])
    
    if MODELS_CONFIG["llama-groq"]["enabled"]:
        ai_clients["llama-groq"] = AsyncOpenAI(api_key=GROQ_API_KEY, base_url=MODELS_CONFIG["llama-groq"]["base_url"])

    # 4. Redis (Async)
    if REDIS_URL:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            logger.info("✅ Redis Connected")
        except:
            logger.warning("❌ Redis Connection Failed")

initialize_systems()

# ===== ASYNC EXTERNAL ENGINE =====

class AsyncResearchEngine:
    """Uses AIOHTTP for parallel, non-blocking lookups"""
    
    @staticmethod
    async def fetch_jisho(session, word):
        try:
            url = f"https://jisho.org/api/v1/search/words?keyword={urllib.parse.quote(word)}"
            async with session.get(url, timeout=1.0) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data['data']:
                        item = data['data'][0]
                        reading = item['japanese'][0].get('reading', '')
                        meaning = item['senses'][0]['english_definitions'][0]
                        return f"- [Jisho] {word}: {reading} ({meaning})"
        except:
            return None
        return None

    @staticmethod
    async def fetch_romajidesu(session, word):
        try:
            url = f"http://www.romajidesu.com/dictionary/meaning-of-{urllib.parse.quote(word)}.html"
            headers = {'User-Agent': 'Mozilla/5.0'}
            async with session.get(url, headers=headers, timeout=1.0) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    soup = BeautifulSoup(text, 'html.parser')
                    romaji = soup.find('div', class_='romaji')
                    if romaji:
                        return f"- [RomajiDesu] {word}: {romaji.text.strip()}"
        except:
            return None
        return None

    @staticmethod
    async def get_research_notes(words: List[str]) -> str:
        """Fetches data for multiple words in PARALLEL"""
        if not words: return ""
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for word in words:
                # Launch both lookups simultaneously
                tasks.append(AsyncResearchEngine.fetch_jisho(session, word))
                # Only check RomajiDesu if it's very complex (optional optimization)
                if len(word) > 1:
                    tasks.append(AsyncResearchEngine.fetch_romajidesu(session, word))
            
            # Wait for all with a hard timeout
            try:
                results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=RESEARCH_TIMEOUT)
                valid_notes = [r for r in results if r]
                return "\n".join(valid_notes)
            except asyncio.TimeoutError:
                logger.warning("Research timed out - proceeding with partial data")
                return ""

# ===== CORE PIPELINE =====

def fast_mecab_convert(text: str) -> Tuple[str, List[str]]:
    """Instant local conversion"""
    if not tagger or not kakasi_converter: return text, []
    
    romaji_parts = []
    complex_words = []
    
    for node in tagger(text):
        word = node.surface
        if not word: continue
        
        # Identify complex words for research
        if any('\u4e00' <= c <= '\u9fff' for c in word):
            complex_words.append(word)
            
        feature = node.feature
        reading = feature[7] if len(feature) > 7 and feature[7] != '*' else None
        
        # Particle rules
        if feature[0] == '助詞':
            if word == 'は': romaji_parts.append('wa')
            elif word == 'へ': romaji_parts.append('e')
            elif word == 'を': romaji_parts.append('wo')
            else: romaji_parts.append(kakasi_converter.do(word).strip())
            continue
            
        if reading:
            romaji_parts.append(kakasi_converter.do(reading).strip())
        else:
            romaji_parts.append(kakasi_converter.do(word).strip())
            
    # Cleanup
    draft = " ".join(romaji_parts).replace("  ", " ").strip()
    return draft, list(set(complex_words)) # deduplicate words

async def ai_validate(model_key: str, japanese: str, draft: str, notes: str):
    """Single AI call wrapper"""
    if model_key not in ai_clients: return None
    
    client = ai_clients[model_key]
    model = MODELS_CONFIG[model_key]
    
    prompt = f"""Task: Correct Romaji.
JAPANESE: {japanese}
DRAFT: {draft}
CONTEXT: {notes}

RULES:
1. Particles: wa/wo/e
2. Long vowels: ō (standard)
3. Return JSON: {{"corrected": "string", "confidence": float}}
"""
    try:
        resp = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model["name"],
            temperature=0.05,
            response_format={"type": "json_object"}
        )
        data = json.loads(resp.choices[0].message.content)
        return {"model": model_key, "data": data}
    except:
        return None

async def process_line(text: str) -> Dict:
    start_ts = time.time()
    
    # 1. CACHE CHECK (Redis -> Memory)
    cache_key = f"romaji:v8:{hashlib.md5(text.encode()).hexdigest()}"
    
    # Check Memory (Fastest)
    if cache_key in memory_cache:
        return memory_cache[cache_key]
        
    # Check Redis (Fast)
    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            data = json.loads(cached)
            memory_cache[cache_key] = data # Populate memory
            return data

    # 2. LOCAL CONVERT (Instant)
    draft_romaji, complex_words = fast_mecab_convert(text)
    
    # 3. RESEARCH (Parallel)
    # We filter to top 3 longest words to save time
    complex_words.sort(key=len, reverse=True)
    targets = complex_words[:3]
    
    research_notes = ""
    if targets:
        research_notes = await AsyncResearchEngine.get_research_notes(targets)

    # 4. AI CONSENSUS (Parallel)
    if ai_clients:
        tasks = [ai_validate(k, text, draft_romaji, research_notes) for k in ai_clients]
        results = await asyncio.gather(*tasks)
        valid = [r for r in results if r]
        
        if valid:
            # Pick best result
            best_res = max(valid, key=lambda x: x['data'].get('confidence', 0) * MODELS_CONFIG[x['model']]['weight'])
            final_romaji = best_res['data']['corrected']
            confidence = best_res['data']['confidence']
        else:
            final_romaji = draft_romaji
            confidence = 0.5
    else:
        final_romaji = draft_romaji
        confidence = 0.5

    # Final formatting
    final_romaji = re.sub(r'\s+', ' ', final_romaji).strip()
    
    result = {
        "original": text,
        "romaji": final_romaji,
        "confidence": confidence,
        "time": round(time.time() - start_ts, 3)
    }
    
    # 5. SAVE TO CACHE
    memory_cache[cache_key] = result
    if redis_client:
        await redis_client.setex(cache_key, 86400 * 7, json.dumps(result)) # 7 Days
        
    return result

# ===== ENDPOINTS =====

@app.get("/")
def home():
    return {"status": "Online", "mode": "Real-Time Music Sync", "cache_size": len(memory_cache)}

@app.get("/convert")
async def convert(text: str):
    if not text: raise HTTPException(400, "Empty text")
    return await process_line(text)

@app.post("/clear-cache")
async def clear_cache(secret: str):
    if secret != "admin123": raise HTTPException(403)
    global memory_cache
    memory_cache = {}
    if redis_client: await redis_client.flushdb()
    return {"status": "cleared"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
