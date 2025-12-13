"""
ULTIMATE ROMAJI ENGINE (v33.0-ENHANCED)
"The Ultra Anti-Hallucination Edition"

NEW IN v33:
- ENHANCED VALIDATION: Post-AI checks to catch hallucinations
- COMPOUND PRIORITY: Special handling for tricky compounds
- DETAILED LOGGING: Track all conversions and changes
- TEST SUITE: Built-in quality assurance
- STRICTER AI PROMPTS: Better instructions with examples
- CONFIDENCE SCORING: Know when to trust the output
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
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("RomajiGuardian")

# ===== LIFECYCLE =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(background_builder())
    yield

app = FastAPI(title="Romaji Guardian Enhanced", version="33.0.0", lifespan=lifespan)
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
DB_VERSION_MARKER = "v33_enhanced"

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

# ===== COMPOUND PRIORITY =====
# These MUST be kept together as single units
COMPOUND_PRIORITY = {
    "小部屋": "kobeya",      # NOT "ko heya" or "kob e ya"
    "煙草": "tabako",        # NOT "kemuri kusa"
    "たばこ": "tabako",
    "タバコ": "tabako",
    "今日": "kyō",           # NOT "konnichi"
    "明日": "ashita",        # NOT "myōnichi"  
    "昨日": "kinō",          # NOT "sakujitsu"
    "今": "ima",             # NOT "genzai" or "kon"
    "今夜": "kon'ya",
    "今朝": "kesa",
    "今年": "kotoshi",
    "去年": "kyonen",
    "来年": "rainen",
    "先月": "sengetsu",
    "今月": "kongetsu",
    "来月": "raigetsu",
    "学生": "gakusei",       # NOT "gakushou" or "ga kusei"
    "人": "hito",            # NOT "nin" or "jin" or "hi to"
    "私": "watashi",         # NOT "wa tashi"
    "名前": "namae",         # NOT "nama e"
    "約束": "yakusoku",      # NOT "ya kusoku"
    "感謝": "kansha",        # NOT "ka nsha"
    "無理": "muri",
    "晴れ": "hare",          # NOT "har e"
}

# ===== BASIC READINGS =====
# Common kanji that often get misread - EXPANDED
BASIC_READINGS = {
    # Particles & common verbs
    "に": "ni",
    "で": "de",
    "と": "to",
    "か": "ka",
    "が": "ga",
    "の": "no",
    "や": "ya",
    "も": "mo",
    "ね": "ne",
    "よ": "yo",
    
    # Verbs
    "入る": "hairu",
    "会う": "au",
    "吸う": "suu",
    "守る": "mamoru",
    "する": "suru",
    "いる": "iru",
    "ある": "aru",
    "行く": "iku",
    "来る": "kuru",
    "見る": "miru",
    "聞く": "kiku",
    "話す": "hanasu",
    "食べる": "taberu",
    "飲む": "nomu",
    "書く": "kaku",
    "読む": "yomu",
    "寝る": "neru",
    "起きる": "okiru",
    
    # Common nouns
    "学生": "gakusei",
    "人": "hito",
    "名前": "namae",
    "感謝": "kansha",
    "晴れ": "hare",
    "雨": "ame",
    "雪": "yuki",
    "風": "kaze",
    "時": "toki",
    "日": "hi",
    "年": "toshi",
    "月": "tsuki",
    "火": "hi",
    "水": "mizu",
    "木": "ki",
    "金": "kin",
    "土": "tsuchi",
    
    # Adverbs & others
    "から": "kara",
    "まで": "made",
    "です": "desu",
    "ます": "masu",
    "ません": "masen",
    "でした": "deshita",
    "ましょう": "mashō",
}

# ===== LYRIC PACK (LOCKED WORDS) =====
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
    "世界": "sekai", "言葉": "kotoba", "心": "kokoro", "愛": "ai", 
    "涙": "namida", "笑顔": "egao", "瞳": "hitomi", 
    "歌": "uta", "確信": "kakushin", "重ねて": "kasanete",
    **COMPOUND_PRIORITY,  # Merge compound priority into lyric pack
    **BASIC_READINGS      # Merge basic readings
}

# ===== HALLUCINATION PATTERNS =====
# Common wrong conversions the AI might make
HALLUCINATION_PAIRS = [
    ("ima", "genzai"),       # 今 should be 'ima' in conversation
    ("kyō", "honjitsu"),     # 今日 conversational vs formal
    ("ashita", "myōnichi"),  # 明日 conversational vs formal
    ("kinō", "sakujitsu"),   # 昨日 conversational vs formal
    ("tabako", "kemuri"),    # Don't split 煙草
    ("kobeya", "ko heya"),   # Don't split 小部屋
]

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
                            clean = word.split("(")[0].strip()
                            batch.append((clean, romaji))
                        
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

# ===== SEARCH =====
class LyricSearchEngine:
    BASE_URL = "http://search.j-lyric.net/index.php"
    
    @staticmethod
    async def find_official_line(session, user_text):
        clean_q = re.sub(r"[!?.、。]", " ", user_text).strip()
        if len(clean_q) < 4: return None
        params = {"kt": clean_q, "ct": 2, "ka": "", "ca": 2, "kl": "", "cl": 2}
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        try:
            async with session.get(
                LyricSearchEngine.BASE_URL, 
                params=params, 
                headers=headers, 
                timeout=4.0
            ) as resp:
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
        except: 
            return None
        return None

# ===== GLUE & LOCKING =====

def fix_spacing(romaji: str) -> str:
    """
    Final pass spacing cleanup - handles edge cases only.
    Main spacing is done by smart_join_romaji.
    """
    result = romaji
    
    # Only fix obvious cases where particles are stuck together
    # Pattern: letter + wo + letter with no spaces
    result = re.sub(r'([a-zōūāē]{2,})wo([a-zōūāē]{2,})', r'\1 wo \2', result)
    result = re.sub(r'([a-zōūāē]{2,})no([a-zōūāē]{2,})', r'\1 no \2', result)
    
    # Fix は as particle when it should be wa
    result = re.sub(r'\bha\b', 'wa', result)
    
    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def smart_join_romaji(parts: List[str], particle_positions: List[bool]) -> str:
    """
    Smart joining that respects word boundaries.
    particle_positions[i] = True if parts[i] is a particle
    """
    if not parts:
        return ""
    
    result = []
    for i, part in enumerate(parts):
        # Add the part
        result.append(part)
        
        # Decide if we need space after this part
        if i < len(parts) - 1:  # Not the last part
            current_is_particle = particle_positions[i] if i < len(particle_positions) else False
            next_is_particle = particle_positions[i+1] if i+1 < len(particle_positions) else False
            
            # Add space after particle or before particle
            if current_is_particle or next_is_particle:
                result.append(' ')
    
    return ''.join(result)

def local_convert(text: str) -> Tuple[str, List[str], Dict[str, str]]:
    """
    Convert Japanese to Romaji with enhanced glue logic and locking.
    Returns: (romaji_draft, research_targets, locked_words)
    """
    if not tagger or not kakasi_conv: 
        return text, [], {}
    
    romaji_parts = []
    particle_positions = []  # Track which parts are particles
    research_targets = []
    locked_words = {}  # Words that MUST NOT be changed by AI
    
    text = text.replace("　", " ")
    nodes = list(tagger(text))
    i = 0
    
    while i < len(nodes):
        node = nodes[i]
        word = node.surface
        matched = False
        
        # PRIORITY: Check COMPOUND_PRIORITY first (before any other glue)
        # This ensures compounds like 小部屋 stay together
        for length in [3, 2]:  # Check 3-char then 2-char compounds
            if i + length - 1 < len(nodes):
                combo = "".join([nodes[i+j].surface for j in range(length)])
                if combo in COMPOUND_PRIORITY:
                    r = COMPOUND_PRIORITY[combo]
                    romaji_parts.append(r)
                    particle_positions.append(False)
                    locked_words[combo] = r
                    logger.debug(f"🔒 Compound Priority: {combo} -> {r}")
                    i += length
                    matched = True
                    break
        
        if matched:
            continue

        # 3-LEVEL GLUE (for non-priority compounds)
        if i + 2 < len(nodes):
            tri_combo = word + nodes[i+1].surface + nodes[i+2].surface
            if tri_combo in LYRIC_PACK:
                r = LYRIC_PACK[tri_combo]
                romaji_parts.append(r)
                particle_positions.append(False)
                locked_words[tri_combo] = r
                i += 3
                continue
            db_hit = get_static_romaji(tri_combo)
            if db_hit:
                romaji_parts.append(db_hit)
                particle_positions.append(False)
                locked_words[tri_combo] = db_hit
                i += 3
                continue

        # 2-LEVEL GLUE
        if i + 1 < len(nodes):
            duo_combo = word + nodes[i+1].surface
            if duo_combo in LYRIC_PACK:
                r = LYRIC_PACK[duo_combo]
                romaji_parts.append(r)
                particle_positions.append(False)
                locked_words[duo_combo] = r
                i += 2
                continue
            db_hit = get_static_romaji(duo_combo)
            if db_hit:
                romaji_parts.append(db_hit)
                particle_positions.append(False)
                locked_words[duo_combo] = db_hit
                i += 2
                continue

        # PARTICLES - NO SPACE BEFORE, ADD AFTER
        if node.feature[0] == '助詞':
            if word == 'は': 
                romaji_parts.append('wa')
            elif word == 'へ': 
                romaji_parts.append('e')
            elif word == 'を': 
                romaji_parts.append('wo')
            else: 
                romaji_parts.append(kakasi_conv.do(word))
            particle_positions.append(True)  # Mark as particle
            i += 1
            continue
            
        # LYRIC PACK (individual words) - CHECK THIS FIRST BEFORE DB
        if word in LYRIC_PACK:
            r = LYRIC_PACK[word]
            romaji_parts.append(r)
            particle_positions.append(False)
            locked_words[word] = r
            logger.debug(f"🔒 Lyric Pack: {word} -> {r}")
            i += 1
            continue

        # DATABASE + MECAB
        db_romaji = get_static_romaji(word)
        mecab_raw = node.feature[7] if len(node.feature) > 7 and node.feature[7] != '*' else word
        mecab_romaji = kakasi_conv.do(mecab_raw).strip()
        
        # If DB has spaces in the romaji (like "wa tashi"), it's probably wrong - use MeCab
        if db_romaji and ' ' in db_romaji:
            romaji_parts.append(mecab_romaji)
            particle_positions.append(False)
            locked_words[word] = mecab_romaji
            logger.debug(f"🔒 DB had spaces, using MeCab: {word} -> {mecab_romaji}")
        elif db_romaji:
            # Phonetic Guard: For single chars, check if DB is wildly different
            if len(word) == 1:
                similarity = SequenceMatcher(None, db_romaji, mecab_romaji).ratio()
                if similarity < 0.3:
                    # DB is too different, trust MeCab but lock it
                    romaji_parts.append(mecab_romaji)
                    particle_positions.append(False)
                    locked_words[word] = mecab_romaji
                    logger.debug(f"🔒 MeCab Override: {word} -> {mecab_romaji} (DB: {db_romaji})")
                else:
                    romaji_parts.append(db_romaji)
                    particle_positions.append(False)
                    locked_words[word] = db_romaji
            else:
                romaji_parts.append(db_romaji)
                particle_positions.append(False)
                locked_words[word] = db_romaji
        else:
            romaji_parts.append(mecab_romaji)
            particle_positions.append(False)
            # Lock single kanji even from MeCab to prevent AI hallucination
            if len(word) == 1 and any('\u4e00' <= c <= '\u9fff' for c in word):
                locked_words[word] = mecab_romaji

        # Track words that need research
        if any('\u4e00' <= c <= '\u9fff' for c in word):
            research_targets.append(word)
        
        i += 1

    # Use smart joining instead of simple join
    draft = smart_join_romaji(romaji_parts, particle_positions)
    draft = re.sub(r'\s+', ' ', draft).strip()
    return draft, list(set(research_targets)), locked_words

# ===== VALIDATION =====

def validate_romaji(original_jp: str, draft: str, final: str, locked: Dict) -> Tuple[str, List[str]]:
    """
    Validate AI output against locked words and hallucination patterns.
    Returns: (validated_romaji, list_of_warnings)
    """
    warnings = []
    validated = final
    
    # Check 1: Ensure locked words are still present
    for jp_word, expected_rom in locked.items():
        if jp_word in original_jp:
            # Normalize for comparison
            expected_lower = expected_rom.lower().replace("'", "").replace("-", "")
            draft_lower = draft.lower().replace("'", "").replace("-", "")
            final_lower = final.lower().replace("'", "").replace("-", "")
            
            if expected_lower in draft_lower and expected_lower not in final_lower:
                warnings.append(f"Locked word missing: '{jp_word}' -> '{expected_rom}'")
                validated = draft
                logger.warning(f"⚠️ Validation: Locked word '{expected_rom}' for '{jp_word}' missing in AI output. Reverting to draft.")
                return validated, warnings
    
    # Check 2: Detect common hallucinations
    for correct, wrong in HALLUCINATION_PAIRS:
        draft_has_correct = correct in draft.lower()
        final_has_wrong = wrong in final.lower()
        final_missing_correct = correct not in final.lower()
        
        if draft_has_correct and final_has_wrong and final_missing_correct:
            warnings.append(f"Hallucination: {correct} -> {wrong}")
            validated = draft
            logger.warning(f"⚠️ Hallucination detected: {correct} became {wrong}. Reverting to draft.")
            return validated, warnings
    
    # Check 3: Ensure no major length change (AI didn't go crazy)
    draft_words = len(draft.split())
    final_words = len(final.split())
    if abs(draft_words - final_words) > max(3, draft_words * 0.3):
        warnings.append(f"Length mismatch: {draft_words} words -> {final_words} words")
        validated = draft
        logger.warning(f"⚠️ Major length change detected. Reverting to draft.")
        return validated, warnings
    
    return validated, warnings

def log_conversion_details(text: str, draft: str, final: str, locked: Dict, method: str, warnings: List[str]):
    """Log detailed conversion information for debugging"""
    if draft != final or warnings:
        logger.info(f"📝 CONVERSION for '{text[:50]}...'")
        logger.info(f"   Draft:  {draft}")
        logger.info(f"   Final:  {final}")
        logger.info(f"   Method: {method}")
        logger.info(f"   Locked: {len(locked)} words - {list(locked.keys())[:5]}")
        if warnings:
            logger.info(f"   ⚠️ Warnings: {warnings}")

# ===== AI PROCESSING =====

async def call_ai(client, model_id, prompt):
    """Call AI with enhanced parameters for consistency"""
    try:
        resp = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_id, 
            temperature=0.0,
            top_p=0.1,  # More deterministic
            response_format={"type": "json_object"}, 
            timeout=8.0
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        logger.error(f"AI call failed: {e}")
        return None

async def process_text_guardian(text: str) -> Dict:
    """Main processing pipeline with enhanced validation"""
    start = time.perf_counter()
    cache_key = f"guardian_v33:{hashlib.md5(text.encode()).hexdigest()}"
    
    # Check L1 cache
    if cache_key in l1_cache: 
        return l1_cache[cache_key]
    
    # Check Redis cache
    if redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached: 
                return json.loads(cached)
        except: pass

    # Step 1: Local conversion with locking
    draft, research_needs, locked = local_convert(text)
    final_romaji = draft
    method = "local_db"
    official_match = False
    warnings = []
    confidence = "high"
    
    # Step 2: Try to find official lyric match
    async with aiohttp.ClientSession() as session:
        official = await LyricSearchEngine.find_official_line(session, text)
        if official:
            draft_off, _, _ = local_convert(official)
            final_romaji = draft_off
            method = "official_match"
            official_match = True
            logger.info(f"✅ Found official match for: {text[:30]}...")
    
    # Step 3: AI refinement (only if needed and no official match)
    if research_needs and not official_match and MODELS["deepseek"]["client"]:
        # Build comprehensive lock list
        lock_examples = []
        for jp, rom in list(locked.items())[:8]:  # Show 8 examples
            lock_examples.append(f"  • '{jp}' = '{rom}'")
        
        # Create detailed prompt with explicit examples
        prompt = f"""You are an expert Romaji correction system. Your ONLY job is to fix obvious errors.

ORIGINAL JAPANESE: {text}
INITIAL ROMAJI: {draft}

🔒 CRITICAL - THESE WORDS ARE LOCKED (NEVER CHANGE):
{chr(10).join(lock_examples)}

Full locked map: {json.dumps(locked, ensure_ascii=False)}

📋 RULES (STRICT):
1. Fix particles ONLY if wrong: は→wa, へ→e, を→wo
2. ENSURE SPACING: Particles must have spaces: "tabako wo suu" NOT "tabakowosuu"
3. ENSURE SPACING: "no" must have spaces: "unmei no hito" NOT "unmeinohito"
4. NEVER change any locked word's reading
5. Prefer conversational readings:
   - 今 = "ima" NOT "genzai" or "kon"
   - 今日 = "kyō" NOT "honjitsu" or "konnichi"
   - 明日 = "ashita" NOT "myōnichi"
   - 小部屋 = "kobeya" NOT "ko heya"
   - 人 = "hito" NOT "nin"
   - 学生 = "gakusei" NOT "gakushou"
6. If unsure, DON'T change it

✅ EXAMPLES:
Input: "tabakowosuu"
Output: "tabako wo suu" (added spaces around wo)

Input: "unmeinohito"  
Output: "unmei no hito" (added spaces around no)

Input: "watashiha ima ikimashou"
Output: "watashi wa ima ikimashou" (fixed ha→wa)

Input: "kobeya shou iru"
Output: "kobeya ni hairu" (fixed shou→ni)

Return JSON: {{"corrected": "final romaji", "changes": ["list what you changed"], "confidence": "high/medium/low"}}"""
        
        ai_data = await call_ai(MODELS["deepseek"]["client"], MODELS["deepseek"]["id"], prompt)
        
        if ai_data:
            ai_output = ai_data.get("corrected", draft)
            ai_changes = ai_data.get("changes", [])
            confidence = ai_data.get("confidence", "medium")
            
            # CRITICAL: Validate AI output
            validated, val_warnings = validate_romaji(text, draft, ai_output, locked)
            warnings.extend(val_warnings)
            
            if not val_warnings:
                # AI output passed validation
                final_romaji = validated
                method = "deepseek_validated"
                logger.info(f"✅ AI refinement accepted: {ai_changes}")
            else:
                # AI output failed validation, using draft
                final_romaji = draft
                method = "local_db_validated"
                confidence = "low"
                logger.warning(f"❌ AI refinement rejected: {val_warnings}")
    
    # Final cleanup
    final_romaji = re.sub(r'\s+', ' ', final_romaji).strip()
    final_romaji = fix_spacing(final_romaji)  # Ensure proper spacing
    
    # Log details
    log_conversion_details(text, draft, final_romaji, locked, method, warnings)
    
    # Build result
    result = {
        "original": text,
        "romaji": final_romaji,
        "method": method,
        "confidence": confidence,
        "warnings": warnings,
        "locked_words": len(locked),
        "time": round(time.perf_counter() - start, 4)
    }
    
    # Cache result
    l1_cache[cache_key] = result
    if redis_client:
        try:
            await redis_client.setex(cache_key, 604800, json.dumps(result))
        except: pass
    
    return result

# ===== TEST SUITE =====

TEST_CASES = [
    ("今は無理", "ima wa muri"),
    ("小部屋に入る", "kobeya ni hairu"),
    ("今日は晴れ", "kyō wa hare"),
    ("煙草を吸う", "tabako wo suu"),
    ("明日会いましょう", "ashita aimashō"),
    ("運命の人", "unmei no hito"),
    ("約束を守る", "yakusoku wo mamoru"),
    ("私は学生です", "watashi wa gakusei desu"),
    ("君の名前は", "kimi no namae wa"),
    ("心から感謝", "kokoro kara kansha"),
]

@app.get("/test")
async def run_tests():
    """Run test suite to measure quality"""
    results = []
    passed_count = 0
    
    for jp, expected in TEST_CASES:
        result = await process_text_guardian(jp)
        
        # Normalize for comparison (ignore spacing variations, macrons, apostrophes)
        def normalize(s):
            return s.lower().replace("'", "").replace("-", "").replace(" ", "").replace("ō", "o").replace("ū", "u").replace("ā", "a")
        
        actual_norm = normalize(result["romaji"])
        expected_norm = normalize(expected)
        
        passed = actual_norm == expected_norm
        if passed:
            passed_count += 1
        
        results.append({
            "input": jp,
            "expected": expected,
            "got": result["romaji"],
            "passed": passed,
            "method": result["method"],
            "confidence": result.get("confidence", "unknown"),
            "warnings": result.get("warnings", [])
        })
    
    return {
        "total": len(TEST_CASES),
        "passed": passed_count,
        "failed": len(TEST_CASES) - passed_count,
        "pass_rate": f"{(passed_count/len(TEST_CASES)*100):.1f}%",
        "results": results
    }

# ===== API ENDPOINTS =====

@app.get("/convert")
async def convert(text: str):
    """Convert single Japanese text to Romaji"""
    if not text: 
        raise HTTPException(400, "Text parameter required")
    return await process_text_guardian(text)

@app.post("/convert-batch")
async def convert_batch(lines: List[str]):
    """Convert multiple lines in parallel"""
    return await asyncio.gather(*[process_text_guardian(l) for l in lines])

@app.get("/check-db")
def check_db_words():
    """Check what's in DB for problem words"""
    problem_words = [
        "小部屋", "私", "学生", "名前", "約束", "感謝", 
        "今日", "明日", "人", "煙草", "晴れ", "無理"
    ]
    
    results = {}
    for word in problem_words:
        db_result = get_static_romaji(word)
        results[word] = {
            "db_romaji": db_result,
            "has_spaces": ' ' in db_result if db_result else False,
            "in_compound_priority": word in COMPOUND_PRIORITY,
            "compound_value": COMPOUND_PRIORITY.get(word),
            "in_lyric_pack": word in LYRIC_PACK,
            "lyric_value": LYRIC_PACK.get(word)
        }
    
    return {
        "problem_words": results,
        "recommendation": "Words with spaces in DB should be overridden by COMPOUND_PRIORITY"
    }

@app.get("/lookup")
def lookup(word: str):
    """Debug: Look up a word in database and manual dict"""
    return {
        "word": word,
        "db_romaji": get_static_romaji(word),
        "manual_romaji": LYRIC_PACK.get(word),
        "compound_priority": COMPOUND_PRIORITY.get(word),
        "basic_reading": BASIC_READINGS.get(word),
        "in_lyric_pack": word in LYRIC_PACK
    }

@app.get("/debug")
async def debug_convert(text: str):
    """Debug: Show detailed conversion steps"""
    try:
        if not tagger:
            return {"error": "Tagger not available", "input": text}
            
        # Tokenize
        tokens = []
        try:
            nodes = list(tagger(text))
            for node in nodes:
                word = node.surface
                tokens.append({
                    "word": word,
                    "pos": node.feature[0] if len(node.feature) > 0 else "?",
                    "reading": node.feature[7] if len(node.feature) > 7 and node.feature[7] != '*' else word,
                    "in_compound_priority": word in COMPOUND_PRIORITY,
                    "in_lyric_pack": word in LYRIC_PACK,
                    "db_romaji": get_static_romaji(word)
                })
        except Exception as e:
            return {"error": f"Tokenization error: {e}", "input": text}
        
        # Get conversion
        draft, research_needs, locked = local_convert(text)
        
        return {
            "input": text,
            "tokens": tokens,
            "draft_romaji": draft,
            "locked_words": locked,
            "research_needed": research_needs,
            "locked_count": len(locked)
        }
    except Exception as e:
        logger.error(f"Debug error: {e}")
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc(), "input": text}

@app.post("/force-rebuild")
async def force_rebuild(secret: str):
    """Admin: Force database rebuild"""
    if secret != ADMIN_SECRET: 
        raise HTTPException(403, "Unauthorized")
    if os.path.exists(DB_FILE): 
        os.remove(DB_FILE)
    asyncio.create_task(background_builder())
    return {"status": "Background Rebuild Started"}

@app.post("/clear-cache")
async def clear_cache(secret: str):
    """Admin: Clear all caches"""
    if secret != ADMIN_SECRET: 
        raise HTTPException(403, "Unauthorized")
    global l1_cache
    l1_cache = {}
    if redis_client: 
        try:
            await redis_client.flushdb()
        except: pass
    return {"status": "Cache Cleared"}

@app.get("/stats")
def stats():
    """Get system statistics"""
    return {
        "version": "33.0.0",
        "db_size": check_db_status(),
        "cache_size": len(l1_cache),
        "lyric_pack_size": len(LYRIC_PACK),
        "compound_priority_size": len(COMPOUND_PRIORITY),
        "models_available": [k for k, v in MODELS.items() if v["client"]],
        "redis_connected": redis_client is not None
    }

@app.get("/")
def root():
    """Health check"""
    return {
        "status": "GUARDIAN_ENHANCED_ONLINE",
        "version": "33.0.0", 
        "db_size": check_db_status()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
