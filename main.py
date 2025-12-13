"""
ULTIMATE ROMAJI ENGINE (v16.0-EXTERNAL-TITAN)
Features:
- LOADS "titan_dictionary.json" (Supports 50,000+ words).
- SINGULARITY MODE: Dictionary enforcement + AI Tribunal.
- ZERO CODE BLOAT: Keeps main.py clean and fast.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
import os
import re
import hashlib
import json
import redis.asyncio as redis
import asyncio
import time
from typing import List, Optional, Dict, Any, Tuple
import logging
import urllib.parse
import aiohttp

# ===== LOGGING & SETUP =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("RomajiTitanExternal")

app = FastAPI(title="Romaji Titan External", version="16.0.0")
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# ===== CONFIGURATION =====
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
REDIS_URL = os.environ.get("REDIS_URL")
ADMIN_SECRET = "admin123"

# SETTINGS
TIMEOUT_RESEARCH = 2.5
TIMEOUT_AI = 6.0
CACHE_TTL = 604800 
MAX_RETRIES = 3

MODELS = {
    "deepseek": {
        "id": "deepseek-chat",
        "client": None,
        "base": "https://api.deepseek.com",
        "key": DEEPSEEK_API_KEY
    },
    "groq": {
        "id": "llama-3.3-70b-versatile",
        "client": None,
        "base": "https://api.groq.com/openai/v1",
        "key": GROQ_API_KEY
    }
}

# ===== GLOBAL STORAGE =====
STATIC_OVERRIDES = {} # Will be loaded from JSON
redis_client = None
tagger = None
kakasi_conv = None
l1_cache = {}

def load_external_dictionary():
    """Loads the massive JSON dictionary if it exists."""
    global STATIC_OVERRIDES
    try:
        if os.path.exists("titan_dictionary.json"):
            with open("titan_dictionary.json", "r", encoding="utf-8") as f:
                STATIC_OVERRIDES = json.load(f)
            logger.info(f"✅ TITAN DICTIONARY LOADED: {len(STATIC_OVERRIDES)} words")
        else:
            logger.warning("⚠️ titan_dictionary.json not found. Using empty dict.")
            STATIC_OVERRIDES = {}
    except Exception as e:
        logger.error(f"❌ Failed to load dictionary: {e}")

def init_globals():
    global tagger, kakasi_conv, redis_client, MODELS
    
    # 1. Load Dictionary
    load_external_dictionary()

    # 2. MeCab
    try:
        import fugashi
        import unidic_lite
        tagger = fugashi.Tagger(f'-d {unidic_lite.DICDIR}')
    except: logger.error("❌ MeCab Failed")

    # 3. Kakasi
    try:
        import pykakasi
        k = pykakasi.kakasi()
        k.setMode("H", "a")
        k.setMode("K", "a")
        k.setMode("J", "a")
        k.setMode("r", "Hepburn")
        kakasi_conv = k.getConverter()
    except: logger.error("❌ Kakasi Failed")

    # 4. AI Clients
    for name, conf in MODELS.items():
        if conf["key"]:
            conf["client"] = AsyncOpenAI(api_key=conf["key"], base_url=conf["base"])

    # 5. Redis
    if REDIS_URL:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            logger.info("✅ Redis Connected")
        except: pass

init_globals()

# ===== RESEARCH ENGINE =====

class ResearchEngine:
    @staticmethod
    async def fetch_jisho(session: aiohttp.ClientSession, word: str) -> Optional[Dict]:
        if word in STATIC_OVERRIDES: return None
        url = f"https://jisho.org/api/v1/search/words?keyword={urllib.parse.quote(word)}"
        try:
            async with session.get(url, timeout=TIMEOUT_RESEARCH) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('data'):
                        for item in data['data']:
                            jp_word = item['japanese'][0].get('word', '')
                            if jp_word == word:
                                reading = item['japanese'][0].get('reading', '')
                                return {"word": word, "reading": reading}
        except: return None
        return None

    @staticmethod
    async def gather_intel(words: List[str]) -> Tuple[str, Dict[str, str]]:
        if not words: return "", {}
        unique = list(set(words))
        
        async with aiohttp.ClientSession() as session:
            tasks = [ResearchEngine.fetch_jisho(session, w) for w in unique]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            notes = []
            forced_readings = {}
            
            for r in results:
                if isinstance(r, dict):
                    notes.append(f"- {r['word']}: {r['reading']}")
                    forced_readings[r['word']] = r['reading']
            
            return "\n".join(notes), forced_readings

# ===== CORE LOGIC =====

def local_convert(text: str) -> Tuple[str, List[str]]:
    if not tagger or not kakasi_conv: return text, []

    romaji_parts = []
    research_targets = []
    text = text.replace("　", " ")
    
    for node in tagger(text):
        word = node.surface
        if not word: continue
        
        # 1. CHECK MASSIVE DICTIONARY
        if word in STATIC_OVERRIDES:
            romaji_parts.append(STATIC_OVERRIDES[word])
            continue
            
        # 2. Particles
        feature = node.feature
        if feature and feature[0] == '助詞':
            if word == 'は': romaji_parts.append('wa')
            elif word == 'へ': romaji_parts.append('e')
            elif word == 'を': romaji_parts.append('wo')
            else: romaji_parts.append(kakasi_conv.do(word))
            continue
            
        # 3. Standard
        reading = feature[7] if len(feature) > 7 and feature[7] != '*' else None
        roma = kakasi_conv.do(reading) if reading else kakasi_conv.do(word)
        romaji_parts.append(roma)

        # 4. Research Unknowns
        if any('\u4e00' <= c <= '\u9fff' for c in word):
            research_targets.append(word)

    draft = re.sub(r'\s+', ' ', " ".join(romaji_parts)).strip()
    return draft, research_targets

async def call_ai_with_retry(client, model_id, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_id,
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=TIMEOUT_AI
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt == MAX_RETRIES - 1: return None
            await asyncio.sleep(0.5)

async def process_text_singularity(text: str) -> Dict:
    start = time.perf_counter()
    
    # 1. CACHE
    cache_key = f"titan_ext:{hashlib.md5(text.encode()).hexdigest()}"
    if cache_key in l1_cache: return l1_cache[cache_key]
    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            l1_cache[cache_key] = json.loads(cached)
            return l1_cache[cache_key]

    # 2. CONVERT
    draft, research_needs = local_convert(text)
    final_romaji = draft
    method = "local_titan"
    
    # 3. AI TRIBUNAL
    if research_needs and MODELS["deepseek"]["client"]:
        notes, forced_map = await ResearchEngine.gather_intel(research_needs)
        
        prompt = f"""Task: Japanese to Romaji.
INPUT: {text}
DRAFT: {draft}
DICTIONARY DATA:
{notes}
RULES: 1. Particles: wa/wo/e. 2. Long vowels: ō.
JSON: {{"corrected": "string"}}
"""
        data = await call_ai_with_retry(MODELS["deepseek"]["client"], MODELS["deepseek"]["id"], prompt)
        if data:
            final_romaji = data.get("corrected", draft)
            method = "deepseek_ai"

    # 4. RESULT
    final_romaji = re.sub(r'\s+', ' ', final_romaji).strip()
    result = {
        "original": text,
        "romaji": final_romaji,
        "method": method,
        "time": round(time.perf_counter()-start, 4)
    }
    
    l1_cache[cache_key] = result
    if redis_client: await redis_client.setex(cache_key, CACHE_TTL, json.dumps(result))
    return result

# ===== ENDPOINTS =====

@app.get("/convert")
async def convert(text: str):
    if not text: raise HTTPException(400, "Text missing")
    return await process_text_singularity(text)

@app.post("/convert-batch")
async def convert_batch(lines: List[str]):
    if not lines: return []
    return await asyncio.gather(*[process_text_singularity(l) for l in lines])

@app.post("/reload-dictionary")
async def reload_dictionary(secret: str):
    """Updates the dictionary without restarting server"""
    if secret != ADMIN_SECRET: raise HTTPException(403)
    load_external_dictionary()
    return {"status": "Reloaded", "words": len(STATIC_OVERRIDES)}

@app.post("/clear-cache")
async def clear_cache(secret: str):
    if secret != ADMIN_SECRET: raise HTTPException(403)
    global l1_cache
    l1_cache = {}
    if redis_client: await redis_client.flushdb()
    return {"status": "Cache Cleared"}

@app.get("/")
def root():
    return {"status": "TITAN_EXTERNAL", "words_loaded": len(STATIC_OVERRIDES)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
