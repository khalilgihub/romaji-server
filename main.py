"""
ULTIMATE ROMAJI ENGINE (v19.0-UNIVERSE)
"The 1 Million Word Edition"

Features:
- UNIVERSE BUILDER: Downloads EDICT (Words) + ENAMDICT (Names).
- TOTAL VOCABULARY: ~950,000 - 1,000,000 entries.
- NAME RECOGNITION: Expert at anime character names and places.
- SINGULARITY MODE: AI Tribunal handles context.
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
import requests
import gzip
import io
import gc # Garbage Collection

# ===== LOGGING & SETUP =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("RomajiUniverse")

app = FastAPI(title="Romaji Universe", version="19.0.0")
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

# ===== UNIVERSE BUILDER (1,000,000 Entries) =====
EDICT_URL = "http://ftp.edrdg.org/pub/Nihongo/edict.gz"
ENAMDICT_URL = "http://ftp.edrdg.org/pub/Nihongo/enamdict.gz"

# Manual Lyrics Pack (Overrides generic dictionary entries)
LYRIC_PACK = {
    "運命": "unmei", "奇跡": "kiseki", "約束": "yakusoku", "記憶": "kioku",
    "物語": "monogatari", "伝説": "densetsu", "永遠": "eien", "瞬間": "shunkan",
    "衝動": "shōdō", "残酷": "zankoku", "天使": "tenshi", "神話": "shinwa",
    "鼓動": "kodō", "旋律": "senritsu", "共鳴": "kyōmei", "幻想": "gensō",
    "楽園": "rakuen", "方舟": "hakobune", "黄昏": "tasogare", "黎明": "reimei",
    "刹那": "setsuna", "悠久": "yūkyū", "彼方": "kanata", "泡沫": "utakata",
    "螺旋": "rasen", "因果": "inga", "輪廻": "rinne", "覚醒": "kakusei",
    "咆哮": "hōkō", "残響": "zankyō", "絶望": "zetsubō", "希望": "kibō",
    "絆": "kizuna", "証": "akashi", "翼": "tsubasa", "扉": "tobira",
    "鍵": "kagi", "鎖": "kusari", "炎": "honō", "氷": "kōri",
    "光": "hikari", "闇": "yami", "影": "kage", "空": "sora",
    "海": "umi", "星": "hoshi", "月": "tsuki", "夢": "yume",
    "私": "watashi", "僕": "boku", "俺": "ore", "君": "kimi", "貴方": "anata"
}

def parse_edrdg_file(url, label, converter, target_dict):
    """Downloads and parses an EDRDG dictionary file (Edict/Enamdict)"""
    try:
        logger.info(f"⬇️ Downloading {label} from {url}...")
        response = requests.get(url, timeout=60)
        
        if response.status_code == 200:
            logger.info(f"📦 Parsing {label}...")
            content = gzip.decompress(response.content).decode("euc-jp", errors="ignore")
            
            count = 0
            for line in content.splitlines():
                try:
                    # Format: KANJI [KANA] /meaning/
                    parts = line.split(" /")
                    header = parts[0]
                    
                    word = ""
                    reading = ""
                    
                    if "[" in header:
                        match = re.match(r"(.+?)\s+\[(.+?)\]", header)
                        if match:
                            word = match.group(1)
                            reading = match.group(2)
                    else:
                        word = header.split()[0]
                        reading = word
                        
                    if word and reading:
                        # Optimization: Filter out extremely long phrases (>12 chars) to save RAM
                        if len(word) < 12:
                            # Direct check to avoid slow pykakasi calls for every line if possible
                            # But we need romaji, so we must convert.
                            # WARNING: 1M conversions is slow. 
                            # We trust PyKakasi speed.
                            romaji = converter.do(reading).strip()
                            target_dict[word] = romaji
                            count += 1
                except: continue
            
            logger.info(f"✅ {label} Parsed: {count} entries added.")
            del content # Free memory immediately
            gc.collect() # Force garbage collection
        else:
            logger.error(f"❌ Failed to download {label}: Status {response.status_code}")
            
    except Exception as e:
        logger.error(f"⚠️ {label} Build Failed: {e}")

def build_universe():
    """Builds the 1 Million Word Dictionary"""
    logger.info("🌌 UNIVERSE: Starting 1,000,000+ Word Build...")
    
    import pykakasi
    k = pykakasi.kakasi()
    k.setMode("H", "a")
    k.setMode("K", "a")
    k.setMode("J", "a")
    k.setMode("r", "Hepburn")
    conv = k.getConverter()
    
    new_db = {}
    
    # 1. Download EDICT (Words) ~180k
    parse_edrdg_file(EDICT_URL, "EDICT (Words)", conv, new_db)
    
    # 2. Download ENAMDICT (Names) ~740k
    # Note: This is huge. We append to the same db.
    parse_edrdg_file(ENAMDICT_URL, "ENAMDICT (Names)", conv, new_db)
    
    # 3. Inject Lyrics (Priority Override)
    new_db.update(LYRIC_PACK)
    
    # 4. Save to Disk
    try:
        logger.info("💾 Saving to disk (this may take a moment)...")
        with open("titan_dictionary.json", "w", encoding="utf-8") as f:
            json.dump(new_db, f, ensure_ascii=False)
        logger.info(f"✅ UNIVERSE BUILD COMPLETE. Total Size: {len(new_db)} words.")
        return new_db
    except Exception as e:
        logger.error(f"❌ Save failed: {e}")
        return LYRIC_PACK

# ===== GLOBAL STORAGE =====
STATIC_OVERRIDES = {} 
redis_client = None
tagger = None
kakasi_conv = None
l1_cache = {}

def load_or_build():
    global STATIC_OVERRIDES
    if os.path.exists("titan_dictionary.json"):
        try:
            logger.info("📂 Loading Dictionary from Disk...")
            with open("titan_dictionary.json", "r", encoding="utf-8") as f:
                STATIC_OVERRIDES = json.load(f)
            
            # If dictionary is too small (<100k), force rebuild to get UNIVERSE
            if len(STATIC_OVERRIDES) < 100000:
                logger.info("⚠️ Dictionary too small. Upgrading to UNIVERSE...")
                STATIC_OVERRIDES = build_universe()
            else:
                logger.info(f"✅ UNIVERSE LOADED: {len(STATIC_OVERRIDES)} entries")
        except:
            STATIC_OVERRIDES = build_universe()
    else:
        STATIC_OVERRIDES = build_universe()

def init_globals():
    global tagger, kakasi_conv, redis_client, MODELS
    
    # 1. Load Dictionary
    load_or_build()

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
            async with session.get(url, timeout=2.5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('data'):
                        for item in data['data']:
                            if item['japanese'][0].get('word') == word:
                                return {"word": word, "reading": item['japanese'][0].get('reading', '')}
        except: return None
        return None

    @staticmethod
    async def gather_intel(words: List[str]) -> Tuple[str, Dict[str, str]]:
        if not words: return "", {}
        unique = list(set(words))
        async with aiohttp.ClientSession() as session:
            tasks = [ResearchEngine.fetch_jisho(session, w) for w in unique]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            notes, forced = [], {}
            for r in results:
                if isinstance(r, dict):
                    notes.append(f"- {r['word']}: {r['reading']}")
                    forced[r['word']] = r['reading']
            return "\n".join(notes), forced

# ===== CORE LOGIC =====

def local_convert(text: str) -> Tuple[str, List[str]]:
    if not tagger or not kakasi_conv: return text, []
    romaji_parts, research_targets = [], []
    text = text.replace("　", " ")
    
    for node in tagger(text):
        word = node.surface
        if not word: continue
        
        # 1. CHECK UNIVERSE DICTIONARY
        if word in STATIC_OVERRIDES:
            romaji_parts.append(STATIC_OVERRIDES[word])
            continue
            
        # 2. Particles
        if node.feature[0] == '助詞':
            if word == 'は': romaji_parts.append('wa')
            elif word == 'へ': romaji_parts.append('e')
            elif word == 'を': romaji_parts.append('wo')
            else: romaji_parts.append(kakasi_conv.do(word))
            continue
            
        # 3. Standard
        reading = node.feature[7] if len(node.feature) > 7 and node.feature[7] != '*' else None
        roma = kakasi_conv.do(reading) if reading else kakasi_conv.do(word)
        romaji_parts.append(roma)

        # 4. Unknowns (Should be nearly zero now)
        if any('\u4e00' <= c <= '\u9fff' for c in word):
            research_targets.append(word)

    draft = re.sub(r'\s+', ' ', " ".join(romaji_parts)).strip()
    return draft, research_targets

async def call_ai(client, model_id, prompt):
    try:
        resp = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_id, temperature=0.0, response_format={"type": "json_object"}, timeout=6.0
        )
        return json.loads(resp.choices[0].message.content)
    except: return None

async def process_text_universe(text: str) -> Dict:
    start = time.perf_counter()
    cache_key = f"universe:{hashlib.md5(text.encode()).hexdigest()}"
    
    if cache_key in l1_cache: return l1_cache[cache_key]
    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            l1_cache[cache_key] = json.loads(cached)
            return l1_cache[cache_key]

    draft, research_needs = local_convert(text)
    final_romaji = draft
    method = "universe_instant"
    
    if research_needs and MODELS["deepseek"]["client"]:
        notes, _ = await ResearchEngine.gather_intel(research_needs)
        prompt = f"JP: {text}\nDraft: {draft}\nDict: {notes}\nRules: wa/wo/e, long vowels ō.\nJSON: {{'corrected': 'string'}}"
        data = await call_ai(MODELS["deepseek"]["client"], MODELS["deepseek"]["id"], prompt)
        if data:
            final_romaji = data.get("corrected", draft)
            method = "deepseek_ai"

    result = {
        "original": text,
        "romaji": re.sub(r'\s+', ' ', final_romaji).strip(),
        "method": method,
        "time": round(time.perf_counter()-start, 4)
    }
    
    l1_cache[cache_key] = result
    if redis_client: await redis_client.setex(cache_key, 604800, json.dumps(result))
    return result

# ===== ENDPOINTS =====

@app.get("/convert")
async def convert(text: str):
    if not text: raise HTTPException(400)
    return await process_text_universe(text)

@app.post("/convert-batch")
async def convert_batch(lines: List[str]):
    return await asyncio.gather(*[process_text_universe(l) for l in lines])

@app.post("/force-rebuild")
async def force_rebuild(secret: str):
    if secret != ADMIN_SECRET: raise HTTPException(403)
    try:
        if os.path.exists("titan_dictionary.json"):
            os.remove("titan_dictionary.json")
        load_or_build()
        return {"status": "UNIVERSE REBUILT", "words": len(STATIC_OVERRIDES)}
    except Exception as e:
        return {"status": "Error", "detail": str(e)}

@app.post("/clear-cache")
async def clear_cache(secret: str):
    if secret != ADMIN_SECRET: raise HTTPException(403)
    global l1_cache
    l1_cache = {}
    if redis_client: await redis_client.flushdb()
    return {"status": "Cache Cleared"}

@app.get("/")
def root():
    return {"status": "UNIVERSE_ONLINE", "words_loaded": len(STATIC_OVERRIDES)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
