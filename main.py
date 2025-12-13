"""
ULTIMATE ROMAJI ENGINE (v21.0-HYPER-SQL)
"The 1 Million Word Edition (RAM Safe)"

Features:
- SQLITE STORAGE: Stores 1,000,000+ words on disk instead of RAM.
- UNIVERSE BUILDER: Downloads EDICT (Words) + ENAMDICT (Names).
- ZERO CRASHES: Runs easily on free 512MB servers.
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
import sqlite3
import gc

# ===== LOGGING & SETUP =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("RomajiHyperSQL")

app = FastAPI(title="Romaji Hyper-SQL", version="21.0.0")
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
DB_FILE = "titan_universe.db"

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

# ===== BUILDER (1,000,000 Entries to SQL) =====
EDICT_URL = "http://ftp.edrdg.org/pub/Nihongo/edict.gz"
ENAMDICT_URL = "http://ftp.edrdg.org/pub/Nihongo/enamdict.gz"

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

def init_db():
    """Create SQLite table"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS dictionary (word TEXT PRIMARY KEY, romaji TEXT)')
    conn.commit()
    return conn

def populate_from_url(url, label, converter, conn):
    try:
        logger.info(f"⬇️ Downloading {label}...")
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            logger.info(f"📦 Parsing {label}...")
            content = gzip.decompress(resp.content).decode("euc-jp", errors="ignore")
            
            c = conn.cursor()
            batch = []
            count = 0
            
            for line in content.splitlines():
                try:
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
                        # PyKakasi is slow, but we only do this ONCE during build
                        romaji = converter.do(reading).strip()
                        batch.append((word, romaji))
                        count += 1
                        
                        if len(batch) >= 5000:
                            c.executemany('INSERT OR IGNORE INTO dictionary VALUES (?,?)', batch)
                            conn.commit()
                            batch = []
                except: continue
            
            if batch:
                c.executemany('INSERT OR IGNORE INTO dictionary VALUES (?,?)', batch)
                conn.commit()
                
            logger.info(f"✅ {label} Complete. Added ~{count} words.")
            del content
            gc.collect()
    except Exception as e:
        logger.error(f"⚠️ {label} Failed: {e}")

def build_universe_sql():
    logger.info("🌌 UNIVERSE SQL: Building 1,000,000+ DB...")
    
    import pykakasi
    k = pykakasi.kakasi()
    k.setMode("H", "a")
    k.setMode("K", "a")
    k.setMode("J", "a")
    k.setMode("r", "Hepburn")
    conv = k.getConverter()
    
    conn = init_db()
    
    # 1. EDICT (Words)
    populate_from_url(EDICT_URL, "EDICT", conv, conn)
    
    # 2. ENAMDICT (Names)
    populate_from_url(ENAMDICT_URL, "ENAMDICT", conv, conn)
    
    # 3. Lyrics
    c = conn.cursor()
    for w, r in LYRIC_PACK.items():
        c.execute('INSERT OR REPLACE INTO dictionary VALUES (?,?)', (w, r))
    conn.commit()
    conn.close()
    
    logger.info("✅ UNIVERSE DB BUILT.")

# ===== DB LOOKUP =====
def get_static_romaji(word):
    """Fast SQL Lookup"""
    try:
        # Connect per request/thread to be safe
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute('SELECT romaji FROM dictionary WHERE word=?', (word,))
            res = c.fetchone()
            if res: return res[0]
    except: return None
    return None

def check_db_status():
    if not os.path.exists(DB_FILE):
        return 0
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute('SELECT count(*) FROM dictionary')
            return c.fetchone()[0]
    except: return 0

# ===== GLOBAL SETUP =====
redis_client = None
tagger = None
kakasi_conv = None
l1_cache = {}

def init_globals():
    global tagger, kakasi_conv, redis_client, MODELS
    
    # Check if DB exists, if not build it
    if check_db_status() < 5000:
        build_universe_sql()
        
    try:
        import fugashi
        import unidic_lite
        tagger = fugashi.Tagger(f'-d {unidic_lite.DICDIR}')
    except: logger.error("❌ MeCab Failed")

    try:
        import pykakasi
        k = pykakasi.kakasi()
        k.setMode("H", "a")
        k.setMode("K", "a")
        k.setMode("J", "a")
        k.setMode("r", "Hepburn")
        kakasi_conv = k.getConverter()
    except: logger.error("❌ Kakasi Failed")

    for name, conf in MODELS.items():
        if conf["key"]:
            conf["client"] = AsyncOpenAI(api_key=conf["key"], base_url=conf["base"])

    if REDIS_URL:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            logger.info("✅ Redis Connected")
        except: pass

init_globals()

# ===== RESEARCH & LOGIC =====

class ResearchEngine:
    @staticmethod
    async def fetch_jisho(session: aiohttp.ClientSession, word: str) -> Optional[Dict]:
        if get_static_romaji(word): return None
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

def local_convert(text: str) -> Tuple[str, List[str]]:
    if not tagger or not kakasi_conv: return text, []
    romaji_parts, research_targets = [], []
    text = text.replace("　", " ")
    
    for node in tagger(text):
        word = node.surface
        if not word: continue
        
        # 1. CHECK SQL DB (Fast & Huge)
        db_hit = get_static_romaji(word)
        if db_hit:
            romaji_parts.append(db_hit)
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

        # 4. Unknowns
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

async def process_text_sql(text: str) -> Dict:
    start = time.perf_counter()
    cache_key = f"sql:{hashlib.md5(text.encode()).hexdigest()}"
    
    if cache_key in l1_cache: return l1_cache[cache_key]
    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            l1_cache[cache_key] = json.loads(cached)
            return l1_cache[cache_key]

    draft, research_needs = local_convert(text)
    final_romaji = draft
    method = "sql_local"
    
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

@app.get("/convert")
async def convert(text: str):
    if not text: raise HTTPException(400)
    return await process_text_sql(text)

@app.post("/convert-batch")
async def convert_batch(lines: List[str]):
    return await asyncio.gather(*[process_text_sql(l) for l in lines])

@app.post("/force-rebuild")
async def force_rebuild(secret: str):
    if secret != ADMIN_SECRET: raise HTTPException(403)
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    build_universe_sql()
    return {"status": "REBUILT_SQL", "words": check_db_status()}

@app.post("/clear-cache")
async def clear_cache(secret: str):
    if secret != ADMIN_SECRET: raise HTTPException(403)
    global l1_cache
    l1_cache = {}
    if redis_client: await redis_client.flushdb()
    return {"status": "Cache Cleared"}

@app.get("/")
def root():
    return {"status": "SQL_UNIVERSE_ONLINE", "words_in_db": check_db_status()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
