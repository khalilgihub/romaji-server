"""
ULTIMATE ROMAJI ENGINE (v28.0-PHONETIC-GUARD)
"The Logic Edition"

Fixes:
- PHONETIC GUARD: Validates DB results against MeCab's official reading.
  (Prevents 'Ima' becoming 'Genzai' or 'Watashi' becoming 'Jibun').
- GLUE LOGIC: Fixes 'Sho'+'Heya' -> 'Kobeya'.
- LYRIC SEARCH: Finds official lyrics.
- RAM SAFE: Uses SQLite.
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
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

# ===== LOGGING & SETUP =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("RomajiPhonetic")

app = FastAPI(title="Romaji Phonetic Guard", version="28.0.0")
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
DB_VERSION_MARKER = "v28_phonetic"

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

# ===== PRIORITY 1: ABSOLUTE OVERRIDES =====
# These skip ALL checks. Use sparingly.
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
    "私": "watashi", "僕": "boku", "俺": "ore", "君": "kimi", "貴方": "anata",
    "明日": "ashita", "今日": "kyō", "昨日": "kinō", "世界": "sekai",
    "言葉": "kotoba", "心": "kokoro", "愛": "ai", "涙": "namida",
    "笑顔": "egao", "瞳": "hitomi", "歌": "uta",
    # Fixes
    "煙草": "tabako", "たばこ": "tabako", "小部屋": "kobeya",
    "今": "ima" # Force Ima
}

# ===== BUILDER =====
EDICT_URL = "http://ftp.edrdg.org/pub/Nihongo/edict.gz"
ENAMDICT_URL = "http://ftp.edrdg.org/pub/Nihongo/enamdict.gz"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS dictionary (word TEXT PRIMARY KEY, romaji TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
    conn.commit()
    return conn

def populate_from_url(url, label, converter, conn):
    try:
        logger.info(f"⬇️ Downloading {label}...")
        resp = requests.get(url, timeout=120) 
        if resp.status_code == 200:
            logger.info(f"📦 Parsing {label}...")
            content = gzip.decompress(resp.content).decode("euc-jp", errors="ignore")
            c = conn.cursor()
            batch = []
            for line in content.splitlines():
                try:
                    parts = line.split(" /")
                    header = parts[0]
                    word, reading = "", ""
                    if "[" in header:
                        match = re.match(r"(.+?)\s+\[(.+?)\]", header)
                        if match: word, reading = match.group(1), match.group(2)
                    else:
                        word, reading = header.split()[0], header.split()[0]
                    
                    if word and reading:
                        if len(word) == 1 and ('\u3040' <= word <= '\u309f'): continue 
                        if any(x in word for x in "▽▼().,"): continue
                        
                        romaji = converter.do(reading).strip()
                        raw_words = word.split(";")
                        for w in raw_words:
                            clean_word = re.sub(r"\(.*?\)", "", w).strip()
                            if clean_word: batch.append((clean_word, romaji))
                        
                        if len(batch) >= 10000:
                            c.executemany('INSERT OR IGNORE INTO dictionary VALUES (?,?)', batch)
                            conn.commit()
                            batch = []
                except: continue
            if batch:
                c.executemany('INSERT OR IGNORE INTO dictionary VALUES (?,?)', batch)
                conn.commit()
            del content
            gc.collect()
            logger.info(f"✅ {label} Done.")
    except Exception as e: logger.error(f"⚠️ {label} Failed: {e}")

def check_db_status():
    if not os.path.exists(DB_FILE): return 0
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT value FROM meta WHERE key='version'")
            ver = c.fetchone()
            if not ver or ver[0] != DB_VERSION_MARKER: return -1
            c.execute('SELECT count(*) FROM dictionary')
            return c.fetchone()[0]
    except: return 0

def build_universe_sql():
    status = check_db_status()
    if status > 50000: return 
    if status == -1 and os.path.exists(DB_FILE): os.remove(DB_FILE)
    logger.info("🌌 STARTING UNIVERSE BUILD...")
    import pykakasi
    k = pykakasi.kakasi()
    k.setMode("H", "a")
    k.setMode("K", "a")
    k.setMode("J", "a")
    k.setMode("r", "Hepburn")
    conv = k.getConverter()
    conn = init_db()
    populate_from_url(EDICT_URL, "EDICT", conv, conn)
    populate_from_url(ENAMDICT_URL, "ENAMDICT", conv, conn)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO meta VALUES ('version', ?)", (DB_VERSION_MARKER,))
    conn.commit()
    conn.close()
    logger.info("✅ DB READY")

def get_static_romaji(word):
    if len(word) < 2: return None 
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute('SELECT romaji FROM dictionary WHERE word=?', (word,))
            res = c.fetchone()
            if res: return res[0]
    except: return None

# ===== INIT =====
redis_client = None
tagger = None
kakasi_conv = None
l1_cache = {}

def init_globals():
    global tagger, kakasi_conv, redis_client, MODELS
    build_universe_sql()
    try:
        import fugashi
        import unidic_lite
        tagger = fugashi.Tagger(f'-d {unidic_lite.DICDIR}')
    except: pass
    try:
        import pykakasi
        k = pykakasi.kakasi()
        k.setMode("H", "a")
        k.setMode("K", "a")
        k.setMode("J", "a")
        k.setMode("r", "Hepburn")
        kakasi_conv = k.getConverter()
    except: pass
    for name, conf in MODELS.items():
        if conf["key"]:
            conf["client"] = AsyncOpenAI(api_key=conf["key"], base_url=conf["base"])
    if REDIS_URL:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        except: pass

init_globals()

# ===== SEARCH =====
class LyricSearchEngine:
    BASE_URL = "http://search.j-lyric.net/index.php"
    @staticmethod
    async def find_official_line(session, user_text):
        clean_q = re.sub(r"[!?.、。]", " ", user_text).strip()
        if len(clean_q) < 4: return None
        params = {"kt": clean_q, "ct": 2, "ka": "", "ca": 2, "kl": "", "cl": 2}
        try:
            async with session.get(LyricSearchEngine.BASE_URL, params=params, timeout=3.0) as resp:
                if resp.status != 200: return None
                soup = BeautifulSoup(await resp.text(), 'html.parser')
                body = soup.find('div', id='mnb')
                if not body: return None
                link = body.find('p', class_='mid').find('a')
                if not link: return None
                async with session.get(link['href'], timeout=3.0) as song_resp:
                    song_soup = BeautifulSoup(await song_resp.text(), 'html.parser')
                    lyric_div = song_soup.find('p', id='Lyric')
                    if lyric_div:
                        for line in lyric_div.get_text(separator="\n").split("\n"):
                            if SequenceMatcher(None, user_text, line).ratio() > 0.6:
                                return line
        except: return None
        return None

# ===== CONVERSION LOGIC (PHONETIC GUARD) =====

def is_phonetically_similar(a, b):
    """Returns True if 'genzai' vs 'ima' are distinct (False)"""
    if not a or not b: return False
    return SequenceMatcher(None, a, b).ratio() > 0.3

def local_convert(text: str) -> Tuple[str, List[str]]:
    if not tagger or not kakasi_conv: return text, []
    romaji_parts = []
    research_targets = []
    
    text = text.replace("　", " ")
    nodes = list(tagger(text))
    
    i = 0
    while i < len(nodes):
        node = nodes[i]
        word = node.surface
        
        # 1. GLUE LOGIC
        if i + 1 < len(nodes):
            next_node = nodes[i+1]
            combined = word + next_node.surface
            if combined in LYRIC_PACK:
                romaji_parts.append(LYRIC_PACK[combined])
                i += 2; continue
            db_hit = get_static_romaji(combined)
            if db_hit:
                # Check Glue Validity
                romaji_parts.append(db_hit)
                i += 2; continue

        # 2. PARTICLE PRIORITY
        if node.feature[0] == '助詞':
            if word == 'は': romaji_parts.append('wa')
            elif word == 'へ': romaji_parts.append('e')
            elif word == 'を': romaji_parts.append('wo')
            else: romaji_parts.append(kakasi_conv.do(word))
            i += 1; continue
            
        # 3. LYRIC PACK PRIORITY
        if word in LYRIC_PACK:
            romaji_parts.append(LYRIC_PACK[word])
            i += 1; continue

        # 4. DATABASE WITH PHONETIC GUARD
        db_romaji = get_static_romaji(word)
        
        # Get standard MeCab reading
        mecab_reading_raw = node.feature[7] if len(node.feature) > 7 and node.feature[7] != '*' else word
        mecab_romaji = kakasi_conv.do(mecab_reading_raw).strip()
        
        if db_romaji:
            # GUARD: If DB result is TOTALLY different from MeCab, assume DB is showing "Meaning" not "Reading"
            # Example: Word="今" -> MeCab="ima" -> DB="genzai". Mismatch! -> Use MeCab.
            if len(word) == 1: # Strict check for single Kanji
                if not is_phonetically_similar(db_romaji, mecab_romaji):
                    # Trust MeCab for single kanji if DB is wild
                    romaji_parts.append(mecab_romaji)
                else:
                    romaji_parts.append(db_romaji)
            else:
                # Trust DB for compounds
                romaji_parts.append(db_romaji)
        else:
            romaji_parts.append(mecab_romaji)

        # 5. RESEARCH TARGET
        if any('\u4e00' <= c <= '\u9fff' for c in word):
            research_targets.append(word)
        i += 1

    draft = re.sub(r'\s+', ' ', " ".join(romaji_parts)).strip()
    return draft, list(set(research_targets))

async def call_ai(client, model_id, prompt):
    try:
        resp = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_id, temperature=0.0, response_format={"type": "json_object"}, timeout=6.0
        )
        return json.loads(resp.choices[0].message.content)
    except: return None

async def process_text_phonetic(text: str) -> Dict:
    start = time.perf_counter()
    cache_key = f"phonetic:{hashlib.md5(text.encode()).hexdigest()}"
    
    if cache_key in l1_cache: return l1_cache[cache_key]
    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached: return json.loads(cached)

    draft, research_needs = local_convert(text)
    final_romaji = draft
    method = "local_db"
    official_match = False
    
    # 1. LYRIC SEARCH
    async with aiohttp.ClientSession() as session:
        official = await LyricSearchEngine.find_official_line(session, text)
        if official:
            draft_off, _ = local_convert(official)
            final_romaji = draft_off
            method = "official_match"
            official_match = True
    
    # 2. AI REFINEMENT
    if (research_needs and not official_match) and MODELS["deepseek"]["client"]:
        prompt = f"Task: Fix Romaji.\nJP: {text}\nDraft: {draft}\nAttention: {', '.join(research_needs)}\nRules: wa/wo/e particles, 'ima' not 'genzai'.\nJSON: {{'corrected': 'string'}}"
        data = await call_ai(MODELS["deepseek"]["client"], MODELS["deepseek"]["id"], prompt)
        if data:
            final_romaji = data.get("corrected", draft)
            method = "deepseek_ai"

    result = {
        "original": text,
        "romaji": re.sub(r'\s+', ' ', final_romaji).strip(),
        "method": method,
        "official_match": official_match,
        "time": round(time.perf_counter()-start, 4)
    }
    
    l1_cache[cache_key] = result
    if redis_client: await redis_client.setex(cache_key, 604800, json.dumps(result))
    return result

@app.get("/convert")
async def convert(text: str):
    if not text: raise HTTPException(400)
    return await process_text_phonetic(text)

@app.post("/convert-batch")
async def convert_batch(lines: List[str]):
    return await asyncio.gather(*[process_text_phonetic(l) for l in lines])

@app.post("/force-rebuild")
async def force_rebuild(secret: str):
    if secret != ADMIN_SECRET: raise HTTPException(403)
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    build_universe_sql()
    return {"status": "REBUILT", "words": check_db_status()}

@app.post("/clear-cache")
async def clear_cache(secret: str):
    if secret != ADMIN_SECRET: raise HTTPException(403)
    global l1_cache
    l1_cache = {}
    if redis_client: await redis_client.flushdb()
    return {"status": "Cache Cleared"}

@app.get("/")
def root():
    return {"status": "PHONETIC_GUARD_ONLINE", "db_size": check_db_status()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
