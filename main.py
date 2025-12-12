from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
import requests
import os
import re
import hashlib
import unicodedata
from typing import List, Optional, Dict, Tuple, Any
import json
import redis
from bs4 import BeautifulSoup
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from fastapi.responses import StreamingResponse
import fugashi  # MeCab wrapper for Python
import pykakasi  # Kana to Romaji converter
import jaconv
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import dataclass
from enum import Enum
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Japanese Lyrics Romaji Converter", 
              description="Convert Japanese lyrics to perfectly spaced Romaji with timestamp alignment")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") 
GENIUS_API_TOKEN = os.environ.get("GENIUS_API_TOKEN")
REDIS_URL = os.environ.get("REDIS_URL")
DEEPSEEK_MODEL = "deepseek-chat"

# --- DATA MODELS ---
@dataclass
class LrcLine:
    timestamp: str
    reference: str

@dataclass
class WordAnalysis:
    surface: str
    reading: Optional[str]
    romaji: Optional[str]
    pos: Optional[str]
    pos_detail: Optional[str]
    base_form: Optional[str]

# --- ENUMS ---
class ProcessingSource(Enum):
    MECAB_ONLY = "MeCab Only"
    MECAB_GENIUS_REFINED = "MeCab + Genius Refined"
    MECAB_AI_REFINED = "MeCab + AI Refined"

# --- GLOBALS ---
client = None
redis_client = None
tagger = None
kakasi_converter = None

# --- CACHES ---
song_cache = {}
line_cache = {}
executor = ThreadPoolExecutor(max_workers=10)

# --- INITIALIZATION ---
def initialize_mecab() -> fugashi.Tagger:
    """Initialize MeCab with proper dictionary"""
    try:
        # Try UniDic first (more accurate)
        try:
            tagger = fugashi.Tagger('-r /dev/null -d /usr/lib/x86_64-linux-gnu/mecab/dic/unidic')
            logger.info("✅ MeCab + UniDic Loaded")
            return tagger
        except Exception:
            # Fallback to IPADIC
            tagger = fugashi.Tagger()
            logger.info("✅ MeCab Loaded (IPADIC)")
            return tagger
    except Exception as e:
        logger.error(f"❌ MeCab failed: {e}")
        return None

def initialize_kakasi():
    """Initialize PyKakasi for romaji conversion"""
    try:
        kakasi = pykakasi.kakasi()
        kakasi.setMode("H", "a")  # Hiragana to ascii
        kakasi.setMode("K", "a")  # Katakana to ascii
        kakasi.setMode("J", "a")  # Japanese (kanji) to ascii
        kakasi.setMode("r", "Hepburn")  # Use Hepburn romanization
        converter = kakasi.getConverter()
        logger.info("✅ PyKakasi Loaded")
        return converter
    except Exception as e:
        logger.error(f"❌ PyKakasi failed: {e}")
        return None

def setup_systems():
    """Initialize all external systems"""
    global client, redis_client, tagger, kakasi_converter
    
    # Initialize NLP tools
    tagger = initialize_mecab()
    kakasi_converter = initialize_kakasi()
    
    # Initialize DeepSeek
    if DEEPSEEK_API_KEY:
        try:
            client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            logger.info(f"✅ DeepSeek AI Online: {DEEPSEEK_MODEL}")
        except Exception as e:
            logger.error(f"❌ DeepSeek AI Failed: {e}")
    
    # Initialize Redis
    if REDIS_URL:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            redis_client.ping()
            logger.info("✅ Redis Online")
        except Exception as e:
            logger.error(f"❌ Redis Failed: {e}")
    
    # Check Genius API
    if GENIUS_API_TOKEN:
        logger.info("✅ Genius API Token Loaded")
    else:
        logger.warning("⚠️ Genius API Token not found (optional)")

setup_systems()

# --- CORE ROMAJI CONVERSION FUNCTIONS ---
def fix_particle_romaji(word: str, romaji: str) -> str:
    """Fix common particle issues in romaji"""
    particle_fixes = {
        "は": "wa",    # Topic particle
        "へ": "e",     # Direction particle
        "を": "wo",    # Direct object particle
        "だ": "da",
        "です": "desu",
        "ます": "masu",
    }
    return particle_fixes.get(word, romaji)

def convert_kana_to_romaji(kana: str) -> str:
    """Convert kana (hiragana/katakana) to romaji"""
    if not kakasi_converter:
        return kana
    
    try:
        # Convert using PyKakasi
        romaji = kakasi_converter.do(kana)
        return romaji
    except Exception as e:
        logger.error(f"Kana conversion error: {e}")
        return kana

def get_word_reading(node: fugashi.Node) -> Optional[str]:
    """Extract reading from MeCab node"""
    if not hasattr(node, 'feature'):
        return None
    
    features = node.feature
    if not features or len(features) < 8:
        return None
    
    # Reading is usually at index 7 in UniDic
    reading = features[7] if features[7] != '*' else None
    
    # If no reading found, try to find kana in other features
    if not reading and features:
        for feat in features:
            if feat and re.match(r'^[\u3040-\u309F\u30A0-\u30FF]+$', feat):
                return feat
    
    return reading

def mecab_to_romaji_perfect(japanese: str) -> str:
    """
    Convert Japanese to perfectly spaced Romaji using MeCab segmentation
    FIXED: Now properly iterates over MeCab nodes
    """
    if not tagger:
        # Fallback: simple conversion without spaces
        if kakasi_converter:
            return kakasi_converter.do(japanese)
        return japanese
    
    try:
        # FIXED: Call tagger directly to get node iterator
        romaji_parts = []
        
        for node in tagger(japanese):  # This is the correct way!
            word = node.surface
            if not word:
                continue
            
            # Get reading from MeCab
            reading = get_word_reading(node)
            
            # Convert to romaji
            if reading:
                # Use the kana reading for conversion
                romaji = convert_kana_to_romaji(reading)
            elif kakasi_converter:
                # Fallback to converting the surface form
                romaji = kakasi_converter.do(word)
            else:
                romaji = word
            
            # Fix common particle issues
            romaji = fix_particle_romaji(word, romaji)
            
            romaji_parts.append(romaji)
        
        # Join with spaces - THIS IS THE KEY FIX
        result = " ".join(romaji_parts)
        
        # Post-processing: clean up spacing
        result = re.sub(r'\s+', ' ', result)  # Normalize multiple spaces
        result = result.strip()
        
        # Special case: fix remaining 'ha' particles
        result = re.sub(r'\bha\b', 'wa', result)
        
        logger.debug(f"Converted '{japanese}' -> '{result}' ({len(romaji_parts)} words)")
        
        return result
        
    except Exception as e:
        logger.error(f"MeCab conversion error: {e}")
        # Fallback to simple conversion
        if kakasi_converter:
            return kakasi_converter.do(japanese)
        return japanese

def mecab_analyze_line(japanese: str) -> List[WordAnalysis]:
    """Detailed analysis of a Japanese line using MeCab"""
    if not tagger:
        return []
    
    try:
        # FIXED: Call tagger directly to get node iterator
        analysis = []
        
        for node in tagger(japanese):  # This is the correct way!
            word = node.surface
            if not word:
                continue
            
            # Get reading
            reading = get_word_reading(node)
            
            # Convert to romaji
            romaji = None
            if reading and kakasi_converter:
                romaji = convert_kana_to_romaji(reading)
                romaji = fix_particle_romaji(word, romaji)
            elif kakasi_converter:
                romaji = kakasi_converter.do(word)
                romaji = fix_particle_romaji(word, romaji)
            
            # Extract POS info
            pos = None
            pos_detail = None
            base_form = None
            
            if hasattr(node, 'feature') and node.feature:
                features = node.feature
                if len(features) > 0:
                    pos = features[0]
                if len(features) > 1:
                    pos_detail = features[1]
                if len(features) > 6:
                    base_form = features[6] if features[6] != '*' else None
            
            analysis.append(WordAnalysis(
                surface=word,
                reading=reading,
                romaji=romaji,
                pos=pos,
                pos_detail=pos_detail,
                base_form=base_form
            ))
        
        return analysis
        
    except Exception as e:
        logger.error(f"MeCab analysis error: {e}")
        return []

# --- HYBRID TRANSLATION SYSTEM ---
async def hybrid_translate_line(japanese: str) -> str:
    """Hybrid approach: MeCab for accuracy + optional AI refinement"""
    # Step 1: Get perfect MeCab romaji with proper spacing
    mecab_romaji = mecab_to_romaji_perfect(japanese)
    
    # Step 2: Optional AI refinement
    if client:
        try:
            analysis = mecab_analyze_line(japanese)
            analysis_str = json.dumps(
                [a.__dict__ for a in analysis], 
                ensure_ascii=False, 
                default=str
            )
            
            prompt = f"""Refine this Romaji translation to sound natural in song lyrics.

ORIGINAL JAPANESE: {japanese}

MECAB ANALYSIS (word-by-word):
{analysis_str}

MECAB ROMAJI (accurate but mechanical): {mecab_romaji}

RULES:
1. Keep the exact meaning
2. Make it flow naturally like song lyrics
3. Preserve proper spacing between words
4. Keep particles: は→wa, を→wo, へ→e
5. Don't change word meanings
6. Output only the refined Romaji

Refined Romaji:"""
            
            completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=DEEPSEEK_MODEL,
                temperature=0.1,
                max_tokens=200
            )
            
            ai_refined = completion.choices[0].message.content.strip()
            
            # Basic validation
            if "を" in japanese and "wo" not in ai_refined.lower():
                return mecab_romaji
            
            return ai_refined
            
        except Exception as e:
            logger.error(f"AI refinement failed: {e}")
    
    return mecab_romaji

# --- LRC PROCESSING ---
async def perfect_align_with_mecab(lrc_lines: List[LrcLine]) -> List[str]:
    """Perfect alignment using MeCab segmentation"""
    logger.info(f"🎯 MeCab Perfect Alignment for {len(lrc_lines)} lines")
    
    aligned = []
    
    for i, lrc_line in enumerate(lrc_lines):
        romaji = await hybrid_translate_line(lrc_line.reference)
        aligned.append(f"{lrc_line.timestamp} {romaji}")
        
        # Progress indicator
        if (i + 1) % 10 == 0 or i == len(lrc_lines) - 1:
            logger.info(f"   Processed {i + 1}/{len(lrc_lines)} lines")
    
    return aligned

# --- GENIUS INTEGRATION ---
async def verify_with_genius(japanese_lines: List[str], genius_romaji: str) -> Dict:
    """Use Genius as verification/reference"""
    if not genius_romaji:
        return {"usable": False, "reason": "No Genius text"}
    
    genius_lines = [l.strip() for l in genius_romaji.split('\n') if l.strip()]
    
    issues = []
    
    # Check for Japanese characters in Genius (should be Romaji)
    jp_chars = sum(len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line)) 
                   for line in genius_lines)
    if jp_chars > len(genius_romaji) * 0.1:
        issues.append("Too many Japanese characters in Genius")
    
    # Check line count
    line_diff = abs(len(genius_lines) - len(japanese_lines))
    if line_diff > max(10, len(japanese_lines) * 0.3):
        issues.append(f"Line count mismatch: {len(genius_lines)} vs {len(japanese_lines)}")
    
    # Check for obvious errors
    error_patterns = [
        (r'\bgenzai\b', '今 should be ima'),
        (r'\bshintai\b', '体 should be karada'),
        (r'\bbakguen\b', 'Probably should be bakuon'),
    ]
    
    for pattern, message in error_patterns:
        if re.search(pattern, genius_romaji, re.IGNORECASE):
            issues.append(message)
    
    is_usable = len(issues) < 2
    
    return {
        "usable": is_usable,
        "issues": issues,
        "line_count": len(genius_lines),
        "genius_lines": genius_lines
    }

# --- SONG PROCESSING PIPELINE ---
async def process_song_ultimate(song: str, artist: str, force_refresh: bool = False) -> Dict[str, Any]:
    """Ultimate processing pipeline"""
    cache_key = f"ultimate:{hashlib.md5(f'{song.lower()}:{artist.lower()}'.encode()).hexdigest()}"
    
    # Check cache first
    if not force_refresh:
        if cache_key in song_cache:
            return song_cache[cache_key]
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                result = json.loads(cached)
                song_cache[cache_key] = result
                return result
    
    logger.info(f"🚀 ULTIMATE Processing: {song} by {artist}")
    start_time = time.time()
    
    try:
        # Step 1: Get LRC timestamps
        lrc_lines = await fetch_lrc_timestamps(song, artist)
        if not lrc_lines:
            raise HTTPException(status_code=404, detail="No lyrics found")
        
        logger.info(f"📝 Found {len(lrc_lines)} timed lines")
        
        # Step 2: Try to get Genius in background
        genius_future = asyncio.create_task(fetch_genius_lyrics_fast(song, artist))
        
        # Step 3: Start MeCab processing
        japanese_lines = [l.reference for l in lrc_lines]
        
        logger.info("🔬 Processing with MeCab...")
        mecab_aligned = await perfect_align_with_mecab(lrc_lines)
        
        # Step 4: Check Genius quality
        genius_result = await genius_future
        genius_info = None
        final_lyrics = mecab_aligned
        source = ProcessingSource.MECAB_ONLY.value
        
        if genius_result:
            romaji_text, _ = genius_result
            genius_info = await verify_with_genius(japanese_lines, romaji_text)
            
            if genius_info["usable"] and len(genius_info.get("issues", [])) == 0:
                logger.info("✨ Genius quality good, using for final polish")
                final_lyrics = await polish_with_genius_reference(mecab_aligned, romaji_text, lrc_lines)
                source = ProcessingSource.MECAB_GENIUS_REFINED.value
            else:
                logger.info(f"⚠️ Genius issues: {genius_info.get('issues', [])}")
        
        # Step 5: Final validation
        validation = validate_final_lyrics(final_lyrics, lrc_lines)
        
        result = {
            "lyrics": '\n'.join(final_lyrics),
            "song": song,
            "artist": artist,
            "source": source,
            "line_count": len(final_lyrics),
            "processing_time": round(time.time() - start_time, 2),
            "validation": validation,
            "cache_key": cache_key,
            "engine": "MeCab+PyKakasi",
            "timestamps_present": True
        }
        
        # Cache result
        if not force_refresh:
            song_cache[cache_key] = result
            if redis_client:
                redis_client.setex(cache_key, 604800, json.dumps(result))
        
        logger.info(f"✅ Completed in {result['processing_time']}s")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

async def polish_with_genius_reference(mecab_lyrics: List[str], genius_romaji: str, 
                                       lrc_lines: List[LrcLine]) -> List[str]:
    """Use Genius as reference to polish MeCab output"""
    if not client:
        return mecab_lyrics
    
    genius_lines = [l.strip() for l in genius_romaji.split('\n') if l.strip()]
    japanese_lines = [l.reference for l in lrc_lines]
    
    prompt = f"""Polish these Romaji lyrics to sound more natural, using Genius as reference.

JAPANESE LINES (first 30):
{chr(10).join([f'{i+1}. {line}' for i, line in enumerate(japanese_lines[:30])])}

MECAB OUTPUT (first 30):
{chr(10).join([f'{i+1}. {line.split(" ", 1)[1] if " " in line else line}' for i, line in enumerate(mecab_lyrics[:30])])}

GENIUS REFERENCE (first 30):
{chr(10).join([f'{i+1}. {line}' for i, line in enumerate(genius_lines[:30])])}

RULES:
1. Preserve MeCab's word accuracy
2. Use Genius for natural phrasing when accurate
3. NEVER use wrong words (e.g., "shintai" for 体, "genzai" for 今)
4. Keep proper spacing between words
5. Output {len(mecab_lyrics)} lines

Output JSON: {{"polished": ["line1", "line2", ...]}}"""

    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(completion.choices[0].message.content)
        polished = data.get("polished", [])
        
        if len(polished) == len(mecab_lyrics):
            # Add timestamps back
            final = []
            for i, line in enumerate(polished):
                if i < len(lrc_lines):
                    timestamp = lrc_lines[i].timestamp
                    final.append(f"{timestamp} {line}")
                else:
                    final.append(line)
            return final
    
    except Exception as e:
        logger.error(f"Polishing failed: {e}")
    
    return mecab_lyrics

def validate_final_lyrics(lyrics: List[str], lrc_lines: List[LrcLine]) -> Dict:
    """Validate final output"""
    issues = []
    
    for i, line in enumerate(lyrics):
        if i >= len(lrc_lines):
            continue
        
        japanese = lrc_lines[i].reference
        romaji_part = line.split(' ', 1)[1] if ' ' in line else line
        
        # Check for critical errors
        if "今" in japanese and "genzai" in romaji_part.lower():
            issues.append(f"Line {i}: Has 'genzai' for 今")
        if "体" in japanese and "shintai" in romaji_part.lower():
            issues.append(f"Line {i}: Has 'shintai' for 体")
        if "を" in japanese and re.search(r'\bo\s+', romaji_part.lower()):
            if "wo" not in romaji_part.lower():
                issues.append(f"Line {i}: Particle を should be 'wo'")
        
        # Check for proper spacing
        if romaji_part and not re.search(r'\s', romaji_part):
            issues.append(f"Line {i}: No spaces in romaji (words may be merged)")
    
    return {
        "total_lines": len(lyrics),
        "issues_found": len(issues),
        "issues": issues[:5] if issues else [],
        "valid": len(issues) == 0
    }

# --- EXTERNAL API FUNCTIONS ---
async def fetch_lrc_timestamps(song: str, artist: str) -> Optional[List[LrcLine]]:
    """Fetch LRC timestamps from LRCLib"""
    try:
        url = "https://lrclib.net/api/get"
        loop = asyncio.get_event_loop()
        
        resp = await loop.run_in_executor(
            None, 
            lambda: requests.get(
                url, 
                params={"track_name": song, "artist_name": artist}, 
                timeout=10
            )
        )
        
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        lrc_text = data.get("syncedLyrics")
        if not lrc_text: 
            return None
        
        lines = []
        for line in lrc_text.split('\n'):
            if not line.strip(): 
                continue
            match = re.match(r'(\[\d+:\d+\.\d+\])\s*(.*)', line)
            if match:
                lines.append(LrcLine(
                    timestamp=match.group(1),
                    reference=match.group(2).strip()
                ))
        return lines
        
    except Exception as e:
        logger.error(f"LRC fetch error: {e}")
        return None

async def fetch_genius_lyrics_fast(song: str, artist: str) -> Optional[Tuple[str, str]]:
    """Fetch lyrics from Genius API"""
    if not GENIUS_API_TOKEN: 
        return None
    
    try:
        headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
        loop = asyncio.get_event_loop()
        
        resp = await loop.run_in_executor(
            None, 
            lambda: requests.get(
                "https://api.genius.com/search", 
                headers=headers, 
                params={"q": f"{song} {artist}"}, 
                timeout=10
            )
        )
        
        data = resp.json()
        if not data['response']['hits']:
            return None
        
        song_url = data['response']['hits'][0]['result']['url']
        
        page = await loop.run_in_executor(
            None,
            lambda: requests.get(song_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        )
        
        soup = BeautifulSoup(page.text, 'html.parser')
        lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
        if not lyrics_divs:
            return None
        
        romaji_text = lyrics_divs[0].get_text(separator='\n', strip=True)
        romaji_text = re.sub(r'\[.*?\]', '', romaji_text)
        romaji_text = re.sub(r'\n\s*\n', '\n', romaji_text)
        romaji_text = romaji_text.strip()
        
        if romaji_text and len(romaji_text) > 50:
            return romaji_text, song_url
        return None
        
    except Exception as e:
        logger.error(f"Genius fetch skipped: {e}")
        return None

# --- FASTAPI ENDPOINTS ---
@app.get("/")
async def root():
    return {
        "status": "Online",
        "version": "MeCab Ultimate v2.1 FIXED",
        "engine": "MeCab+PyKakasi+AI",
        "accuracy": "Word-perfect segmentation with proper spacing",
        "fix_applied": "Corrected MeCab node iteration for proper word spacing",
        "features": [
            "Perfect word spacing in romaji",
            "MeCab-based accurate segmentation",
            "AI refinement for natural flow",
            "Genius verification",
            "LRC timestamp alignment"
        ],
        "endpoints": {
            "/convert": "Simple Japanese to Romaji (with spaces)",
            "/convert_mecab": "Advanced MeCab conversion with analysis",
            "/analyze": "Detailed word-by-word analysis",
            "/get_song_ultimate": "Complete lyrics processing",
            "/stream_mecab": "Real-time streaming",
            "/test_spacing": "Test word spacing accuracy",
            "/health": "System health check"
        }
    }

@app.get("/convert")
async def convert_simple(text: str = "") -> Dict:
    """Simple conversion endpoint with proper spacing"""
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    cache_key = f"simple:{hashlib.md5(text.encode()).hexdigest()}"
    if cache_key in line_cache:
        return {"original": text, "romaji": line_cache[cache_key]}
    
    romaji = mecab_to_romaji_perfect(text)
    line_cache[cache_key] = romaji
    
    return {
        "original": text,
        "romaji": romaji,
        "word_count": len(romaji.split()),
        "has_spaces": " " in romaji,
        "engine": "MeCab+PyKakasi"
    }

@app.get("/convert_mecab")
async def convert_mecab(text: str = "") -> Dict:
    """MeCab-based conversion with detailed analysis"""
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    cache_key = f"mecab:{hashlib.md5(text.encode()).hexdigest()}"
    if cache_key in line_cache:
        cached = line_cache[cache_key]
        return {
            "original": text,
            "romaji": cached["romaji"],
            "analysis": cached.get("analysis", []),
            "engine": "MeCab+PyKakasi"
        }
    
    romaji = mecab_to_romaji_perfect(text)
    analysis = mecab_analyze_line(text)
    
    result = {
        "original": text,
        "romaji": romaji,
        "analysis": [a.__dict__ for a in analysis],
        "word_count": len(analysis),
        "has_spaces": " " in romaji,
        "engine": "MeCab+PyKakasi"
    }
    
    line_cache[cache_key] = result
    return result

@app.get("/analyze")
async def analyze_text(text: str = "") -> Dict:
    """Detailed MeCab analysis"""
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    analysis = mecab_analyze_line(text)
    romaji = mecab_to_romaji_perfect(text)
    
    return {
        "text": text,
        "romaji": romaji,
        "analysis": [a.__dict__ for a in analysis],
        "word_count": len(analysis),
        "engine": "MeCab+PyKakasi",
        "has_spaces": " " in romaji
    }

@app.get("/get_song_ultimate")
async def get_song_ultimate(song: str, artist: str, force_refresh: bool = False) -> Dict:
    """Ultimate accuracy endpoint for complete song processing"""
    return await process_song_ultimate(song, artist, force_refresh)

@app.get("/stream_mecab")
async def stream_mecab(song: str, artist: str):
    """Real-time streaming with MeCab"""
    async def generate():
        yield json.dumps({"status": "starting", "song": song, "artist": artist}) + "\n"
        
        lrc_lines = await fetch_lrc_timestamps(song, artist)
        if not lrc_lines:
            yield json.dumps({"error": "No lyrics found"}) + "\n"
            return
        
        yield json.dumps({"status": "lrc_loaded", "count": len(lrc_lines)}) + "\n"
        
        # Stream with MeCab
        for i, lrc_line in enumerate(lrc_lines):
            romaji = mecab_to_romaji_perfect(lrc_line.reference)
            line = f"{lrc_line.timestamp} {romaji}"
            
            yield json.dumps({
                "line": line,
                "index": i,
                "total": len(lrc_lines),
                "progress": (i + 1) / len(lrc_lines),
                "engine": "MeCab",
                "word_count": len(romaji.split()),
                "has_spaces": " " in romaji
            }) + "\n"
            
            await asyncio.sleep(0.01)
        
        yield json.dumps({"status": "complete"}) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.
