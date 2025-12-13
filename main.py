"""
ULTIMATE ROMAJI ENGINE (v30.0-HYPER-GLUE)
"The Multi-Part Fixer"

Fixes:
- 3-LEVEL GLUE: Stitches words split into 3 parts (e.g. Ko+Be+Ya).
- USER-AGENT: Prevents Lyric Search from getting blocked.
- DEBUG TRACE: Tells you exactly which method was used for each line.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
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
from contextlib import asynccontextmanager

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("RomajiHyperGlue")

# ===== LIFECYCLE =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(background_builder())
    yield

app = FastAPI(title="Romaji Hyper-Glue", version="30.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# ===== CONFIG =====
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
REDIS_URL = os.environ.get("REDIS_URL")
ADMIN_SECRET = "admin123"
DB_FILE = "titan_universe.db"
DB_VERSION_MARKER = "v30_glue"

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

# ===== MANUAL OVERRIDES =====
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
    "笑顔": "egao", "瞳": "hitomi", "煙草": "tabako", "歌": "uta", 
    "小部屋": "kobeya", "今": "ima"
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

async def download_and_parse_stream(url, label, converter, conn):
    try:
        temp_file = f"temp_{label}.gz"
        logger.info(f"⬇️ {label}: Downloading...")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    with open(temp_file, 'wb') as f:
                        while True:
                            chunk = await resp.content.read(1024*1024)
                            if not chunk: break
                            f.write(chunk)
                else: return

        logger.info(f"📦 {label}: Parsing...")
        c = conn.cursor()
        batch = []
        
        with gzip.open(temp_file, 'rt', encoding='euc-jp', errors='ignore') as f:
            for line in f:
                try:
                    if " [" in line:
                        parts = line.split(" [", 1)
                        word, rest = parts[0], parts[1]
                        reading = rest.split("]", 1)[0] if "]" in rest else word
                    else:
                        word = line.split(" /", 1)[0]
                        reading = word

                    if word and reading:
                        if len(word) == 1 and ('\u3040' <= word <= '\u309f'): continue 
                        if any(x in word for x in "▽▼().,"): continue
                        
                        romaji = converter.do(reading).strip()
                        if ";" in word:
                            for w in word.split(";"):
                                clean = w.split("(")[0].strip()
                                if clean: batch.append((clean, romaji))
                        else:
                            batch.append((word, romaji))
                        
                        if len(batch) >= 20000:
                            c.executemany('INSERT OR IGNORE INTO dictionary VALUES (?,?)', batch)
                            conn.commit()
                            batch = []
                            await asyncio.sleep(0.01)
                except: continue
        
        if batch:
            c.executemany('INSERT OR IGNORE INTO dictionary VALUES (?,?)', batch)
            conn.commit()
        
        if os.path.exists(temp_file): os.remove(temp_file)
        logger.info(f"✅ {label}: Done.")
        
    except Exception as e:
        logger.error(f"⚠️ {label} Error: {e}")

async def background_builder():
    if check_db_status() > 50000: return
    logger.info("🚀 BUILDER STARTED")
    import pykakasi
    k = pykakasi.kakasi()
    k.setMode("H", "a")
    k.setMode("K", "a")
    k.setMode("J", "a")
    k.setMode("r", "Hepburn")
    conv = k.getConverter()
    
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    conn = init_db()
    
    await download_and_parse_stream(EDICT_URL, "EDICT", conv, conn)
    await download_and_parse_stream(ENAMDICT_URL, "ENAMDICT", conv, conn)
    
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO meta VALUES ('version', ?)", (DB_VERSION_MARKER,))
    conn.commit()
    conn.close()
    logger.info("🎉 BUILD COMPLETE")

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

# ===== LYRIC SEARCH =====
class LyricSearchEngine:
    BASE_URL = "http://search.j-lyric.net/index.php"
    @staticmethod
    async def find_official_line(session, user_text):
        clean_q = re.sub(r"[!?.、。]", " ", user_text).strip()
        if len(clean_q) < 4: return None
        params = {"kt": clean_q, "ct": 2, "ka": "", "ca": 2, "kl": "", "cl": 2}
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        try:
            async with session.get(LyricSearchEngine.BASE_URL, params=params, headers=headers, timeout=4.0) as resp:
                if resp.status != 200: return None
                soup = BeautifulSoup(await resp.text(), 'html.parser')
                body = soup.find('div', id='mnb')
                if not body: return None
                link = body.find('p', class_='mid').find('a')
                if not link: return None
                
                async with session.get(link['href'], headers=headers, timeout=4.0) as song_resp:
                    song_soup = BeautifulSoup(await song_resp.text(), 'html.parser')
                    lyric_div = song_soup.find('p', id='Lyric')
                    if lyric_div:
                        for line in lyric_div.get_text(separator="\n").split("\n"):
                            if SequenceMatcher(None, user_text, line).ratio() > 0.6:
                                return line
        except: return None
        return None

def is_phonetically_similar(a, b):
    if not a or not b: return False
    return SequenceMatcher(None, a, b).ratio() > 0.3

# ===== CONVERSION (HYPER GLUE) =====

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
        
        # --- 3-LEVEL GLUE LOGIC ---
        # Try gluing 3 parts: A + B + C (e.g., Ko+Be+Ya)
        if i + 2 < len(nodes):
            tri_combo = word + nodes[i+1].surface + nodes[i+2].surface
            if tri_combo in LYRIC_PACK:
                romaji_parts.append(LYRIC_PACK[tri_combo])
                i += 3; continue
            db_hit = get_static_romaji(tri_combo)
            if db_hit:
                romaji_parts.append(db_hit)
                i += 3; continue

        # Try gluing 2 parts: A + B (e.g. En+Sou)
        if i + 1 < len(nodes):
            duo_combo = word + nodes[i+1].surface
            if duo_combo in LYRIC_PACK:
                romaji_parts.append(LYRIC_PACK[duo_combo])
                i += 2; continue
            db_hit = get_static_romaji(duo_combo)
            if db_hit:
                romaji_parts.append(db_hit)
                i += 2; continue
        # --------------------------

        # Particles
        if node.feature[0] == '助詞':
            if word == 'は': romaji_parts.append('wa')
            elif word == 'へ': romaji_parts.append('e')
            elif word == 'を': romaji_parts.append('wo')
            else: romaji_parts.append(kakasi_conv.do(word))
            i += 1; continue
            
        # Lyric Pack
        if word in LYRIC_PACK:
            romaji_parts.append(LYRIC_PACK[word])
            i += 1; continue

        # DB Check
        db_romaji = get_static_romaji(word)
        mecab_raw = node.feature[7] if len(node.feature) > 7 and node.feature[7] != '*' else word
        mecab_romaji = kakasi_conv.do(mecab_raw).strip()
        
        if db_romaji:
            if len(word) == 1:
                if not is_phonetically_similar(db_romaji, mecab_romaji):
                    romaji_parts.append(mecab_romaji) 
                else:
                    romaji_parts.append(db_romaji)
            else:
                romaji_parts.append(db_romaji)
        else:
            romaji_parts.append(mecab_romaji)

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

async def process_text_glue(text: str) -> Dict:
    start = time.perf_counter()
    cache_key = f"glue_v30:{hashlib.md5(text.encode()).hexdigest()}"
    
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
        prompt = f"Task: Fix Romaji.\nJP: {text}\nDraft: {draft}\nAttention: {', '.join(research_needs)}\nRules: wa/wo/e particles.\nJSON: {{'corrected': 'string'}}"
        data = await call_ai(MODELS["deepseek"]["client"], MODELS["deepseek"]["id"], prompt)
        if data:
            final_romaji = data.get("corrected", draft)
            method = "deepseek_ai"

    result = {
        "original": text,
        "romaji": re.sub(r'\s+', ' ', final_romaji).strip(),
        "method": method,
        "trace": f"Official:{official_match}, DB_Size:{check_db_status()}",
        "time": round(time.perf_counter()-start, 4)
    }
    
    l1_cache[cache_key] = result
    if redis_client: await redis_client.setex(cache_key, 604800, json.dumps(result))
    return result

@app.get("/convert")
async def convert(text: str):
    if not text: raise HTTPException(400)
    return await process_text_glue(text)

@app.post("/convert-batch")
async def convert_batch(lines: List[str]):
    return await asyncio.gather(*[process_text_glue(l) for l in lines])

@app.post("/force-rebuild")
async def force_rebuild(secret: str):
    if secret != ADMIN_SECRET: raise HTTPException(403)
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    asyncio.create_task(background_builder())
    return {"status": "Background Rebuild Started"}

@app.post("/clear-cache")
async def clear_cache(secret: str):
    if secret != ADMIN_SECRET: raise HTTPException(403)
    global l1_cache
    l1_cache = {}
    if redis_client: await redis_client.flushdb()
    return {"status": "Cache Cleared"}

@app.get("/")
def root():
    return {"status": "HYPER_GLUE_ONLINE", "db_size": check_db_status()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
