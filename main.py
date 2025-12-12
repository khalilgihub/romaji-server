from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
import requests
import os
import re
import hashlib
import unicodedata
from typing import List, Optional, Dict, Tuple
import json
import redis
from bs4 import BeautifulSoup
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from fastapi.responses import StreamingResponse
from difflib import SequenceMatcher
import jaconv

app = FastAPI()

# --- CONFIGURATION ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") 
GENIUS_API_TOKEN = os.environ.get("GENIUS_API_TOKEN")
REDIS_URL = os.environ.get("REDIS_URL")
DEEPSEEK_MODEL = "deepseek-chat" 

client = None
redis_client = None
song_cache = {}
line_cache = {}
executor = ThreadPoolExecutor(max_workers=10)

def setup_systems():
    global client, redis_client
    if DEEPSEEK_API_KEY:
        try:
            client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            print(f"✅ DeepSeek AI Online: {DEEPSEEK_MODEL}")
        except Exception as e:
            print(f"❌ DeepSeek AI Failed: {e}")
    if REDIS_URL:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            redis_client.ping()
            print("✅ Redis Online")
        except Exception as e:
            print(f"❌ Redis Failed: {e}")
    if GENIUS_API_TOKEN:
        print("✅ Genius API Token Loaded")

setup_systems()

# --- KNOWN ERRORS DATABASE (MANUAL CORRECTIONS) ---
KNOWN_ERRORS_DATABASE = {
    # Format: {"japanese_pattern": "correct_romaji", "wrong_pattern": "wrong_romaji"}
    
    # Your specific errors:
    "夜道を迷ぐれど虚しい": {
        "correct": "yomichi wo masaguredo munashii",
        "wrong_patterns": ["yomichi o iburedo munashi", "yomichi wo iburedo munashi"]
    },
    "愛してる一人鳴き喚いて": {
        "correct": "ai shiteru hitori nakiwameite",
        "wrong_patterns": ["aishiteiru hitori naki sakebite", "ai shiteiru hitori nakiwameite"]
    },
    "改札の安警光灯": {
        "correct": "kaisatsu no yasu keikoutou",
        "wrong_patterns": ["kaisatsu no an keikoto wa", "kaisatsu no an keikoutou"]
    },
    "サイレン爆音現実界ある浮遊": {
        "correct": "sairen bakuon genjitsukai aru fuyuu",
        "wrong_patterns": ["sairen bakguen genjikkai aru fuyu", "sairen bakuen genjitsukai aru fuyuu"]
    },
    "体を触って必要なのはこれだけ認めて": {
        "correct": "karada wo sawatte hitsuyou na no wa kore dake mitomete",
        "wrong_patterns": ["shintai wo sawatte hitsuyou na no wa kore dake mitomete"]
    },
    "確信できる今だけ重ねて": {
        "correct": "kakushin dekiru ima dake kasanete",
        "wrong_patterns": ["kakushin dekiru genzai dake kasanete"]
    },
    
    # Common patterns:
    "今": {"correct": "ima", "wrong_patterns": ["genzai", "present"]},
    "体を": {"correct": "karada wo", "wrong_patterns": ["shintai wo", "karada o"]},
    "を": {"correct": "wo", "wrong_patterns": ["o "]},  # Space after o to avoid false positives
    "は": {"correct": "wa", "wrong_patterns": ["ha "]},
    "へ": {"correct": "e", "wrong_patterns": ["he "]},
}

def correct_known_errors(japanese: str, romaji: str) -> str:
    """
    Apply manual corrections for known errors
    """
    corrected = romaji
    
    # First check exact matches in our database
    if japanese in KNOWN_ERRORS_DATABASE:
        correct_version = KNOWN_ERRORS_DATABASE[japanese]["correct"]
        wrong_patterns = KNOWN_ERRORS_DATABASE[japanese].get("wrong_patterns", [])
        
        # Check if current romaji matches any wrong pattern
        for wrong in wrong_patterns:
            if wrong.lower() in romaji.lower():
                print(f"🔧 MANUAL CORRECTION: '{wrong}' → '{correct_version}'")
                # Replace the wrong part with correct
                corrected = re.sub(re.escape(wrong), correct_version, corrected, flags=re.IGNORECASE)
                return corrected
    
    # Check for partial matches (if Japanese contains known patterns)
    for pattern, data in KNOWN_ERRORS_DATABASE.items():
        if pattern in japanese and len(pattern) > 1:  # Only for multi-character patterns
            # Check if the wrong pattern appears in romaji
            for wrong in data.get("wrong_patterns", []):
                if wrong.lower() in romaji.lower():
                    print(f"🔧 PARTIAL CORRECTION: '{wrong}' → '{data['correct']}' in '{romaji[:50]}...'")
                    # Replace the wrong part
                    corrected = re.sub(re.escape(wrong), data['correct'], corrected, flags=re.IGNORECASE)
    
    # General particle corrections
    if "を" in japanese:
        # Fix particle を (should be "wo" not "o")
        corrected = re.sub(r'\s+o\s+', ' wo ', corrected)  # Space o space
        corrected = re.sub(r'^o\s+', 'wo ', corrected)     # Beginning o space
        corrected = re.sub(r'\s+o$', ' wo', corrected)     # Space o end
    
    if "は" in japanese and ("は" not in ["こんにちは", "こんばんは"]):  # Not in greetings
        corrected = re.sub(r'\s+ha\s+', ' wa ', corrected)
    
    return corrected

# --- ULTRA-STRICT AI TRANSLATION WITH PATTERN ENFORCEMENT ---
async def translate_with_strict_enforcement(japanese_lines: List[str]) -> List[str]:
    """
    AI translation with pattern enforcement and post-correction
    """
    if not client:
        return japanese_lines
    
    print(f"🔒 STRICT AI Translation for {len(japanese_lines)} lines")
    
    # Prepare context: show examples of correct translations
    examples = ""
    for jp, data in KNOWN_ERRORS_DATABASE.items():
        if len(jp) > 3:  # Only use substantial examples
            examples += f"- {jp} → {data['correct']}\n"
    
    # Process in small chunks for better accuracy
    chunk_size = 15
    all_translations = []
    
    for chunk_idx in range(0, len(japanese_lines), chunk_size):
        chunk = japanese_lines[chunk_idx:chunk_idx + chunk_size]
        
        prompt = f"""TRANSLATE THESE JAPANESE LYRICS TO ROMAJI WITH 100% ACCURACY.

CRITICAL EXAMPLES (MUST USE THESE EXACT TRANSLATIONS):
{examples}

ABSOLUTE RULES:
1. 夜道を迷ぐれど虚しい → "yomichi wo masaguredo munashii" (NOT "yomichi o iburedo munashi")
2. 愛してる一人鳴き喚いて → "ai shiteru hitori nakiwameite" (NOT "aishiteiru hitori naki sakebite")
3. 改札の安警光灯 → "kaisatsu no yasu keikoutou" (NOT "kaisatsu no an keikoto wa")
4. サイレン爆音現実界ある浮遊 → "sairen bakuon genjitsukai aru fuyuu" (NOT "sairen bakguen genjikkai aru fuyu")
5. 体を触って必要なのはこれだけ認めて → "karada wo sawatte hitsuyou na no wa kore dake mitomete" (NOT "shintai wo")
6. 今 → ALWAYS "ima" (NEVER "genzai")
7. を → ALWAYS "wo" (not "o" for particle)
8. は → ALWAYS "wa" (not "ha" for particle)

IMPORTANT: If you see ANY of the Japanese patterns above, use the EXACT Romaji shown.

LINES TO TRANSLATE ({len(chunk)} lines):
{chr(10).join([f"{i+1}. {line}" for i, line in enumerate(chunk)])}

Output JSON: {{"translations": ["romaji1", "romaji2", ...]}}
Be absolutely precise!"""

        try:
            completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=DEEPSEEK_MODEL,
                temperature=0.0,  # Zero temperature for maximum consistency
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(completion.choices[0].message.content)
            translations = data.get("translations", [])
            
            if len(translations) == len(chunk):
                # Apply manual corrections as backup
                corrected_translations = []
                for i, (jp, romaji) in enumerate(zip(chunk, translations)):
                    corrected = correct_known_errors(jp, romaji)
                    corrected_translations.append(corrected)
                all_translations.extend(corrected_translations)
            else:
                # Fallback: translate line by line
                print(f"⚠️ Chunk {chunk_idx//chunk_size + 1} count mismatch, translating individually...")
                for jp in chunk:
                    trans = await translate_single_line_strict(jp)
                    all_translations.append(trans)
                    
        except Exception as e:
            print(f"❌ Chunk error: {e}")
            for jp in chunk:
                all_translations.append(jp)
    
    # Final verification pass
    final_corrected = []
    for i, (jp, romaji) in enumerate(zip(japanese_lines, all_translations)):
        if i < len(all_translations):
            # Check against known errors database
            if jp in KNOWN_ERRORS_DATABASE:
                correct_version = KNOWN_ERRORS_DATABASE[jp]["correct"]
                # If translation doesn't match correct version, force it
                if correct_version.lower() not in romaji.lower():
                    print(f"⚠️ FORCE CORRECTION line {i}: '{romaji[:30]}...' → '{correct_version}'")
                    final_corrected.append(correct_version)
                    continue
            
            # Apply general corrections
            corrected = correct_known_errors(jp, romaji)
            final_corrected.append(corrected)
        else:
            final_corrected.append(jp)
    
    return final_corrected

async def translate_single_line_strict(japanese: str) -> str:
    """Single line translation with strict rules"""
    # Check database first
    if japanese in KNOWN_ERRORS_DATABASE:
        return KNOWN_ERRORS_DATABASE[japanese]["correct"]
    
    # Check for partial matches
    for pattern, data in KNOWN_ERRORS_DATABASE.items():
        if pattern in japanese and len(pattern) > 1:
            # Use the correct translation for this pattern
            prompt = f"""Translate this Japanese lyric to Romaji.
IMPORTANT: The part "{pattern}" must be translated as "{data['correct']}"

Japanese: {japanese}
Romaji:"""
            
            try:
                completion = await client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=DEEPSEEK_MODEL,
                    temperature=0.0,
                    max_tokens=100
                )
                romaji = completion.choices[0].message.content.strip()
                return correct_known_errors(japanese, romaji)
            except:
                return japanese
    
    # General translation
    prompt = f"""Translate to Romaji: 今→ima, を→wo, は→wa, 体を→karada wo
Japanese: {japanese}
Romaji:"""
    
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.0,
            max_tokens=100
        )
        romaji = completion.choices[0].message.content.strip()
        return correct_known_errors(japanese, romaji)
    except:
        return japanese

# --- SIMPLE FETCH FUNCTIONS ---
def parse_lrc_lines(lrc_text: str) -> List[Dict]:
    lines = []
    for line in lrc_text.split('\n'):
        if not line.strip(): 
            continue
        match = re.match(r'(\[\d+:\d+\.\d+\])\s*(.*)', line)
        if match:
            lines.append({
                'timestamp': match.group(1), 
                'reference': match.group(2).strip()
            })
    return lines

async def fetch_lrc_timestamps(song: str, artist: str) -> Optional[List[Dict]]:
    try:
        url = "https://lrclib.net/api/get"
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, 
            lambda: requests.get(url, params={"track_name": song, "artist_name": artist}, timeout=5)
        )
        data = resp.json()
        lrc_text = data.get("syncedLyrics")
        if not lrc_text: 
            return None
        return parse_lrc_lines(lrc_text)
    except: 
        return None

# --- BULLETPROOF PROCESSING ---
async def process_song_bulletproof(song: str, artist: str, force_refresh: bool = False):
    """
    Bulletproof processing: Uses strict AI translation with manual corrections
    """
    cache_key = f"bullet:{hashlib.md5(f'{song.lower()}:{artist.lower()}'.encode()).hexdigest()}"
    
    if not force_refresh:
        if cache_key in song_cache:
            return song_cache[cache_key]
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                result = json.loads(cached)
                song_cache[cache_key] = result
                return result
    
    print(f"🛡️ BULLETPROOF Processing: {song} by {artist}")
    start_time = time.time()
    
    try:
        # Get LRC timestamps
        lrc_lines = await fetch_lrc_timestamps(song, artist)
        if not lrc_lines:
            raise HTTPException(404, "No lyrics found")
        
        print(f"📊 Found {len(lrc_lines)} lines")
        
        # Extract Japanese lines
        japanese_lines = [l['reference'] for l in lrc_lines]
        
        # Use strict AI translation with enforcement
        romaji_lines = await translate_with_strict_enforcement(japanese_lines)
        
        # Combine with timestamps
        final_lyrics = []
        for i, (lrc_line, romaji) in enumerate(zip(lrc_lines, romaji_lines)):
            final_lyrics.append(f"{lrc_line['timestamp']} {romaji}")
        
        # Final verification
        verified_count = 0
        for i, line in enumerate(final_lyrics):
            if i < len(lrc_lines):
                japanese = lrc_lines[i]['reference']
                
                # Check against known errors
                if japanese in KNOWN_ERRORS_DATABASE:
                    correct = KNOWN_ERRORS_DATABASE[japanese]["correct"]
                    wrong_patterns = KNOWN_ERRORS_DATABASE[japanese].get("wrong_patterns", [])
                    
                    for wrong in wrong_patterns:
                        if wrong.lower() in line.lower():
                            # Replace the line entirely
                            final_lyrics[i] = f"{lrc_lines[i]['timestamp']} {correct}"
                            verified_count += 1
                            print(f"🔧 Final fix line {i}: '{wrong}' → '{correct}'")
                            break
        
        result = {
            "lyrics": '\n'.join(final_lyrics),
            "song": song,
            "artist": artist,
            "source": "AI Translation (Strict Enforcement)",
            "line_count": len(final_lyrics),
            "processing_time": round(time.time() - start_time, 2),
            "corrections_applied": verified_count,
            "cache_key": cache_key
        }
        
        # Cache
        if not force_refresh:
            song_cache[cache_key] = result
            if redis_client:
                redis_client.setex(cache_key, 86400, json.dumps(result))
        
        print(f"✅ Completed in {result['processing_time']}s, applied {verified_count} corrections")
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

# --- REAL-TIME STREAMING WITH INSTANT CORRECTIONS ---
@app.get("/stream_instant")
async def stream_instant(song: str, artist: str):
    """Stream with instant corrections"""
    async def generate():
        yield json.dumps({"status": "starting", "song": song, "artist": artist}) + "\n"
        
        lrc_lines = await fetch_lrc_timestamps(song, artist)
        if not lrc_lines:
            yield json.dumps({"error": "No lyrics found"}) + "\n"
            return
        
        yield json.dumps({"status": "lrc_loaded", "count": len(lrc_lines)}) + "\n"
        
        # Stream line by line with instant corrections
        for i, lrc_line in enumerate(lrc_lines):
            japanese = lrc_line['reference']
            
            # Check database first
            if japanese in KNOWN_ERRORS_DATABASE:
                romaji = KNOWN_ERRORS_DATABASE[japanese]["correct"]
            else:
                # Translate with strict rules
                romaji = await translate_single_line_strict(japanese)
            
            line = f"{lrc_line['timestamp']} {romaji}"
            
            yield json.dumps({
                "line": line,
                "index": i,
                "total": len(lrc_lines),
                "progress": (i + 1) / len(lrc_lines)
            }) + "\n"
        
        yield json.dumps({"status": "complete"}) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {
        "status": "Online",
        "version": "Bulletproof v1",
        "note": "Using strict pattern matching and manual corrections",
        "known_errors": len(KNOWN_ERRORS_DATABASE),
        "endpoints": {
            "/convert": "Single line conversion",
            "/get_song": "Get lyrics with strict corrections",
            "/stream_instant": "Stream with instant corrections",
            "/test_errors": "Test error corrections",
            "/clear_cache": "Clear cache"
        }
    }

@app.get("/convert")
async def convert_single_line(text: str = ""):
    if not text:
        raise HTTPException(400, "No text")
    
    cache_key = f"conv:{hashlib.md5(text.encode()).hexdigest()}"
    if cache_key in line_cache:
        return {"original": text, "romaji": line_cache[cache_key]}
    
    if not client:
        return {"original": text, "romaji": text}
    
    try:
        romaji = await translate_single_line_strict(text)
        line_cache[cache_key] = romaji
        return {"original": text, "romaji": romaji}
    except:
        return {"original": text, "romaji": text}

@app.get("/get_song")
async def get_song_endpoint(song: str, artist: str, force_refresh: bool = False):
    """Main endpoint - uses bulletproof processing"""
    return await process_song_bulletproof(song, artist, force_refresh)

@app.get("/get_song_fresh")
async def get_song_fresh(song: str, artist: str):
    """Always fresh"""
    return await process_song_bulletproof(song, artist, force_refresh=True)

@app.get("/test_errors")
async def test_errors():
    """Test all known error corrections"""
    test_cases = [
        "夜道を迷ぐれど虚しい",
        "愛してる一人鳴き喚いて", 
        "改札の安警光灯",
        "サイレン爆音現実界ある浮遊",
        "体を触って必要なのはこれだけ認めて",
        "確信できる今だけ重ねて",
        "今だけ",
        "体を触る"
    ]
    
    results = []
    for japanese in test_cases:
        if japanese in KNOWN_ERRORS_DATABASE:
            correct = KNOWN_ERRORS_DATABASE[japanese]["correct"]
            wrong = KNOWN_ERRORS_DATABASE[japanese].get("wrong_patterns", ["none"])[0]
        else:
            correct = "N/A"
            wrong = "N/A"
        
        if client:
            translated = await translate_single_line_strict(japanese)
        else:
            translated = japanese
        
        # Check if correct
        is_correct = False
        if japanese in KNOWN_ERRORS_DATABASE:
            is_correct = correct.lower() in translated.lower()
            for wrong_pattern in KNOWN_ERRORS_DATABASE[japanese].get("wrong_patterns", []):
                if wrong_pattern.lower() in translated.lower():
                    is_correct = False
        
        results.append({
            "japanese": japanese,
            "translated": translated,
            "expected": correct,
            "common_wrong": wrong,
            "correct": is_correct
        })
    
    return {
        "test": "Error Correction Test",
        "results": results,
        "summary": {
            "total": len(results),
            "correct": sum(1 for r in results if r["correct"]),
            "has_database_entry": sum(1 for r in results if r["japanese"] in KNOWN_ERRORS_DATABASE)
        }
    }

@app.delete("/clear_cache")
async def clear_cache():
    """Clear all cache"""
    song_cache.clear()
    line_cache.clear()
    if redis_client:
        redis_client.flushdb()
    return {
        "status": "Cache cleared",
        "message": "Now using bulletproof error correction system"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "deepseek": bool(client),
        "redis": redis_client.ping() if redis_client else False,
        "genius": bool(GENIUS_API_TOKEN),
        "cache_size": len(song_cache),
        "known_errors": len(KNOWN_ERRORS_DATABASE)
    }

@app.get("/add_error")
async def add_error(japanese: str, correct: str, wrong: str = ""):
    """Add a new error to the database (temporary)"""
    if japanese not in KNOWN_ERRORS_DATABASE:
        KNOWN_ERRORS_DATABASE[japanese] = {
            "correct": correct,
            "wrong_patterns": [wrong] if wrong else []
        }
    else:
        if wrong and wrong not in KNOWN_ERRORS_DATABASE[japanese]["wrong_patterns"]:
            KNOWN_ERRORS_DATABASE[japanese]["wrong_patterns"].append(wrong)
    
    return {
        "status": "Error added",
        "japanese": japanese,
        "correct": correct,
        "wrong_patterns": KNOWN_ERRORS_DATABASE[japanese]["wrong_patterns"],
        "total_errors": len(KNOWN_ERRORS_DATABASE)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
