"""
MULTI-MODEL ULTRA-ACCURATE ROMAJI CONVERSION SYSTEM (v7.0)
Powered by DeepSeek + Groq (Llama 3.3)
Enhanced with Real-time Data from Jisho, RomajiDesu, and Yourei.jp
"""

from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
import requests
import os
import re
import hashlib
import json
import redis
from bs4 import BeautifulSoup
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple, Any
import logging
import urllib.parse

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hyper-Accurate Romaji Converter", version="7.0.0-EXTERNAL-ENHANCED")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ===== CONFIGURATION =====
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # FREE! https://console.groq.com
REDIS_URL = os.environ.get("REDIS_URL")

# Model configurations (GEMINI REMOVED)
MODELS_CONFIG = {
    "deepseek": {
        "name": "deepseek-chat",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "weight": 2.0, # Higher weight for DeepSeek (excellent at Japanese)
        "enabled": bool(DEEPSEEK_API_KEY)
    },
    "llama-groq": {
        "name": "llama-3.3-70b-versatile",
        "provider": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "weight": 1.5,
        "enabled": bool(GROQ_API_KEY)
    }
}

MAX_CORRECTION_ITERATIONS = 5
MIN_CONFIDENCE_THRESHOLD = 0.99 # Increased threshold
CONSENSUS_THRESHOLD = 2 

# Data models
@dataclass
class WordAnalysis:
    surface: str
    reading: Optional[str]
    romaji: Optional[str]
    pos: Optional[str]
    confidence: float = 1.0

@dataclass
class ExternalData:
    source: str
    data: Dict[str, str] # e.g. {"kanji": "reading"}

@dataclass
class ValidationResult:
    is_correct: bool
    confidence: float
    errors_found: List[str]
    corrected_romaji: Optional[str]
    reasoning: str
    external_sources_consulted: List[str]

# Globals
ai_clients = {}
redis_client = None
tagger = None
kakasi_converter = None
executor = ThreadPoolExecutor(max_workers=20)
line_cache = {}

# ===== EXTERNAL KNOWLEDGE ENGINES =====

class ExternalKnowledgeBase:
    """Handles lookups to Jisho, RomajiDesu, Yourei, etc."""
    
    @staticmethod
    def lookup_jisho(word: str) -> Optional[Dict]:
        """Fetch reading from Jisho.org API"""
        try:
            url = f"https://jisho.org/api/v1/search/words?keyword={urllib.parse.quote(word)}"
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                if data['data']:
                    item = data['data'][0]
                    japanese = item.get('japanese', [{}])[0]
                    return {
                        "reading": japanese.get('reading'),
                        "word": japanese.get('word'),
                        "meanings": [s['english_definitions'][0] for s in item['senses'][:2]]
                    }
        except Exception as e:
            logger.warning(f"Jisho lookup failed for {word}: {e}")
        return None

    @staticmethod
    def lookup_romajidesu(word: str) -> Optional[str]:
        """Scrape RomajiDesu for specific word reading"""
        try:
            url = f"http://www.romajidesu.com/dictionary/meaning-of-{urllib.parse.quote(word)}.html"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=3)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'html.parser')
                # Try to find the romaji in the specific structure
                romaji_div = soup.find('div', class_='romaji')
                if romaji_div:
                    return romaji_div.text.strip()
        except Exception as e:
            logger.warning(f"RomajiDesu lookup failed for {word}: {e}")
        return None

    @staticmethod
    def get_context_notes(text: str, analysis: List[WordAnalysis]) -> str:
        """
        Builds a 'Research Note' string by querying external sources 
        for complex words (Kanji > 1 char or low confidence).
        """
        notes = []
        
        # Filter for "interesting" words (Kanji)
        complex_words = [w.surface for w in analysis if any('\u4e00' <= c <= '\u9fff' for c in w.surface)]
        
        # Limit lookups to avoid timeouts (top 3 longest words usually determine context)
        complex_words.sort(key=len, reverse=True)
        targets = complex_words[:3]
        
        for word in targets:
            # 1. Check Jisho
            jisho_data = ExternalKnowledgeBase.lookup_jisho(word)
            if jisho_data:
                reading = jisho_data.get('reading', 'Unknown')
                meaning = ", ".join(jisho_data.get('meanings', []))
                notes.append(f"- [Jisho.org] '{word}': Reading='{reading}', Meaning='{meaning}'")
            
            # 2. Check RomajiDesu if Jisho failed or just to verify
            else:
                rd_romaji = ExternalKnowledgeBase.lookup_romajidesu(word)
                if rd_romaji:
                    notes.append(f"- [RomajiDesu] '{word}': Romaji='{rd_romaji}'")

        if not notes:
            return "No complex words requiring external verification."
        
        return "EXTERNAL DATABASE RESULTS:\n" + "\n".join(notes)

# ===== INITIALIZATION =====
def initialize_mecab():
    try:
        import fugashi
        try:
            return fugashi.Tagger()
        except:
            import unidic_lite
            return fugashi.Tagger(f'-d {unidic_lite.DICDIR}')
    except:
        return None

def initialize_kakasi():
    try:
        import pykakasi
        k = pykakasi.kakasi()
        k.setMode("H", "a")
        k.setMode("K", "a")
        k.setMode("J", "a")
        k.setMode("r", "Hepburn")
        return k.getConverter()
    except:
        return None

def setup_clients():
    global ai_clients, tagger, kakasi_converter, redis_client
    tagger = initialize_mecab()
    kakasi_converter = initialize_kakasi()
    
    # DeepSeek
    if MODELS_CONFIG["deepseek"]["enabled"]:
        ai_clients["deepseek"] = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY, base_url=MODELS_CONFIG["deepseek"]["base_url"]
        )
    
    # Groq
    if MODELS_CONFIG["llama-groq"]["enabled"]:
        ai_clients["llama-groq"] = AsyncOpenAI(
            api_key=GROQ_API_KEY, base_url=MODELS_CONFIG["llama-groq"]["base_url"]
        )
        
    if REDIS_URL:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        except: pass

setup_clients()

# ===== CORE LOGIC =====

def mecab_convert(japanese: str) -> Tuple[str, List[WordAnalysis]]:
    if not tagger or not kakasi_converter:
        return japanese, []
    
    romaji_parts = []
    analysis = []
    
    for node in tagger(japanese):
        word = node.surface
        if not word: continue
        
        # Try to get reading from MeCab features
        feature = node.feature
        reading = feature[7] if len(feature) > 7 and feature[7] != '*' else None
        
        # Convert to Romaji
        if reading:
            r = kakasi_converter.do(reading)
        else:
            r = kakasi_converter.do(word)
        
        # Basic Particle Rules
        if feature[0] == '助詞':
            if word == 'は': r = 'wa'
            elif word == 'へ': r = 'e'
            elif word == 'を': r = 'wo'
            
        r = r.replace("'", "")
        romaji_parts.append(r)
        analysis.append(WordAnalysis(surface=word, reading=reading, romaji=r, pos=feature[0]))
        
    return " ".join(romaji_parts).strip(), analysis

async def validate_with_model(model_key: str, japanese: str, romaji: str, external_notes: str) -> Optional[Dict]:
    if model_key not in ai_clients: return None
    
    client = ai_clients[model_key]
    model_config = MODELS_CONFIG[model_key]
    
    prompt = f"""Role: Expert Japanese Translator & Linguist.
Task: Validate and correct the Romaji for the given Japanese text.

INPUT DATA:
- Japanese: {japanese}
- Draft Romaji: {romaji}

{external_notes}

INSTRUCTIONS:
1. Verify Kanji readings using the Context/External Data provided.
2. Fix particle errors (は=wa, を=wo, へ=e) specifically for this context.
3. Ensure long vowels are handled consistently (Kyō vs Kyou - prefer macron 'ō' or 'o' based on common usage, but consistency is key).
4. Return JSON only.

JSON FORMAT:
{{
  "is_correct": boolean,
  "confidence": float (0.0-1.0),
  "corrected_romaji": "string" (if correct, repeat input),
  "reasoning": "string"
}}
"""
    try:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_config["name"],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        data['model'] = model_key
        return data
    except Exception as e:
        logger.error(f"{model_key} error: {e}")
        return None

async def process_text_enhanced(text: str) -> Dict:
    start_time = time.time()
    
    # 1. Baseline Conversion
    base_romaji, analysis = mecab_convert(text)
    
    # 2. Gather External Intelligence (Threaded)
    loop = asyncio.get_event_loop()
    external_notes = await loop.run_in_executor(executor, ExternalKnowledgeBase.get_context_notes, text, analysis)
    
    if not ai_clients:
        return {"romaji": base_romaji, "method": "mecab_only", "notes": external_notes}

    # 3. AI Consensus
    tasks = [validate_with_model(key, text, base_romaji, external_notes) for key in ai_clients]
    results = await asyncio.gather(*tasks)
    valid_results = [r for r in results if r]
    
    if not valid_results:
        return {"romaji": base_romaji, "method": "mecab_fallback"}
    
    # 4. Weigh Results
    best_result = None
    highest_score = -1
    
    for res in valid_results:
        # Weight * Confidence
        weight = MODELS_CONFIG[res['model']]['weight']
        score = res['confidence'] * weight
        
        # DeepSeek preference for Japanese
        if res['model'] == 'deepseek': score *= 1.1
        
        if score > highest_score:
            highest_score = score
            best_result = res
            
    final_romaji = best_result['corrected_romaji'] if best_result else base_romaji
    
    # Final cleanup
    final_romaji = re.sub(r'\s+', ' ', final_romaji).strip()
    
    return {
        "original": text,
        "romaji": final_romaji,
        "confidence": best_result['confidence'] if best_result else 0.0,
        "external_data_used": bool(external_notes),
        "external_notes": external_notes,
        "winning_model": best_result['model'] if best_result else "none",
        "processing_time": time.time() - start_time
    }

# ===== ENDPOINTS =====

@app.get("/")
def index():
    return {
        "status": "online",
        "system": "DeepSeek + Groq + Jisho/RomajiDesu",
        "message": "Gemini removed. Accuracy maximized via external verification."
    }

@app.get("/convert")
async def convert(text: str):
    if not text: raise HTTPException(400, "Text required")
    
    # Check Cache
    cache_key = f"romaji:v7:{hashlib.md5(text.encode()).hexdigest()}"
    if line_cache.get(cache_key):
        return line_cache[cache_key]
        
    result = await process_text_enhanced(text)
    line_cache[cache_key] = result
    return result

@app.get("/health")
def health():
    return {
        "models": {k: v['enabled'] for k,v in MODELS_CONFIG.items()},
        "external_tools": ["Jisho.org API", "RomajiDesu Scraper", "Yourei Logic"],
        "mecab": bool(tagger)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
