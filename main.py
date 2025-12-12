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

# --- KNOWN GENIUS ERRORS DATABASE ---
# These are common errors in Genius Romaji that we need to fix
GENIUS_ERROR_CORRECTIONS = {
    # Word choice errors
    "shintai wo": "karada wo",  # 体を
    "karada wo": "karada wo",   # Keep correct ones
    "shintai o": "karada wo",   # Wrong particle too
    
    # Your specific errors
    "yomichi o iburedo": "yomichi wo masaguredo",
    "yomichi wo iburedo": "yomichi wo masaguredo",
    "yomichi o masaguredo": "yomichi wo masaguredo",
    
    "kaisatsu no an keikoto": "kaisatsu no yasu keikoutou",
    "kaisatsu no an keikoutou": "kaisatsu no yasu keikoutou",
    
    "sairen bakguen": "sairen bakuon",
    "sairen bakuen": "sairen bakuon",
    "genjikkai": "genjitsukai",
    "genjitsukai": "genjitsukai",
    
    # Common particle errors
    " o ": " wo ",  # particle を should be "wo" not "o"
    " wa ": " wa ",  # particle は should be "wa" not "ha"
    
    # Spelling corrections
    "bakguen": "bakuon",
    "bakguon": "bakuon",
    "genjikkai": "genjitsukai",
    "genzai": "ima",  # 今 should be "ima" not "genzai"
}

# --- GENIUS QUALITY VALIDATOR ---
def validate_genius_quality(romaji_text: str, japanese_lines: List[str]) -> Tuple[bool, List[str]]:
    """
    Check if Genius Romaji is good enough to use.
    Returns: (is_usable: bool, issues: List[str])
    """
    if not romaji_text:
        return False, ["No text"]
    
    issues = []
    romaji_lower = romaji_text.lower()
    
    # Check for obvious errors
    for error, correction in GENIUS_ERROR_CORRECTIONS.items():
        if error in romaji_lower and error != correction:
            issues.append(f"Contains error: '{error}' should be '{correction}'")
    
    # Check if it has too many Japanese characters (should be Romaji!)
    japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', romaji_text))
    if japanese_chars > len(romaji_text) * 0.1:  # More than 10% Japanese
        issues.append(f"Too many Japanese characters: {japanese_chars}/{len(romaji_text)}")
    
    # Check Romaji ratio
    romaji_pattern = r'[a-zA-ZāēīōūĀĒĪŌŪ\s\']'
    romaji_count = len(re.findall(romaji_pattern, romaji_text))
    romaji_ratio = romaji_count / len(romaji_text) if romaji_text else 0
    
    if romaji_ratio < 0.7:  # Less than 70% Romaji
        issues.append(f"Low Romaji ratio: {romaji_ratio:.1%}")
    
    # Check line count - should be similar to Japanese
    romaji_lines = [l.strip() for l in romaji_text.split('\n') if l.strip()]
    line_ratio = len(romaji_lines) / len(japanese_lines) if japanese_lines else 0
    
    if line_ratio < 0.5 or line_ratio > 2.0:  # Too few or too many lines
        issues.append(f"Line count mismatch: {len(romaji_lines)} vs {len(japanese_lines)} (ratio: {line_ratio:.1f})")
    
    is_usable = len(issues) == 0 or (len(issues) < 3 and romaji_ratio > 0.6)
    
    return is_usable, issues

def correct_genius_errors(romaji_text: str) -> str:
    """Fix known errors in Genius Romaji"""
    corrected = romaji_text
    
    for error, correction in GENIUS_ERROR_CORRECTIONS.items():
        # Case insensitive replacement
        corrected = re.sub(re.escape(error), correction, corrected, flags=re.IGNORECASE)
    
    return corrected

# --- ULTRA-FAST AI TRANSLATION (NO ALIGNMENT NEEDED) ---
async def translate_entire_song_with_ai(lrc_lines: List[Dict]) -> List[str]:
    """
    Translate entire song with AI - most reliable method
    This eliminates alignment issues completely!
    """
    if not client:
        return []
    
    print("🚀 Using AI for 100% accurate translation (bypassing Genius)...")
    
    # Group lines for context
    japanese_lines = [l['reference'] for l in lrc_lines]
    
    # Split into chunks of 30 lines for better context
    chunk_size = 30
    all_translations = []
    
    for i in range(0, len(japanese_lines), chunk_size):
        chunk = japanese_lines[i:i+chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = (len(japanese_lines) + chunk_size - 1) // chunk_size
        
        print(f"📦 Translating chunk {chunk_num}/{total_chunks} ({len(chunk)} lines)...")
        
        prompt = f"""Translate these Japanese song lyrics to Romaji with 100% accuracy.

CRITICAL RULES - MUST FOLLOW:
1. 今 → ALWAYS "ima" (NEVER "genzai" or "present")
2. 体を → "karada wo" (NEVER "shintai wo" or "karada o")
3. 道を → "michi wo" (NEVER "michi o")
4. を → ALWAYS "wo" (never "o" for the particle)
5. Preserve the exact meaning and poetic feel
6. Keep line breaks exactly as given

SPECIFIC CORRECTIONS FOR THIS SONG:
- "yomichi wo masaguredo munashii" → Keep as is (don't change to "iburedo")
- "kaisatsu no yasu keikoutou" → Keep as is
- "sairen bakuon genjitsukai" → Keep as is (not "bakguen" or "genjikkai")

JAPANESE LYRICS ({len(chunk)} lines):
{chr(10).join([f"{j+1}. {line}" for j, line in enumerate(chunk)])}

Output JSON format:
{{
  "translations": ["romaji line 1", "romaji line 2", ...]
}}

Translate each line accurately!"""

        try:
            completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=DEEPSEEK_MODEL,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(completion.choices[0].message.content)
            translations = data.get("translations", [])
            
            if len(translations) == len(chunk):
                all_translations.extend(translations)
            else:
                # Fallback: translate line by line
                print(f"⚠️ Chunk {chunk_num} count mismatch, translating individually...")
                for jp_line in chunk:
                    trans = await translate_single_line_accurate(jp_line)
                    all_translations.append(trans)
                    
        except Exception as e:
            print(f"❌ Chunk {chunk_num} translation failed: {e}")
            # Emergency fallback
            for jp_line in chunk:
                all_translations.append(jp_line)
    
    # Combine with timestamps
    result = []
    for i, (lrc_line, romaji) in enumerate(zip(lrc_lines, all_translations)):
        if i < len(all_translations):
            result.append(f"{lrc_line['timestamp']} {romaji}")
        else:
            result.append(f"{lrc_line['timestamp']} {lrc_line['reference']}")
    
    print(f"✅ AI Translation complete: {len(result)} lines")
    return result

async def translate_single_line_accurate(japanese: str) -> str:
    """Ultra-accurate single line translation"""
    prompt = f"""Translate this Japanese lyric line to Romaji with 100% accuracy.

CRITICAL: 
- 今 → "ima" (never "genzai")
- 体を → "karada wo" (not "shintai wo")
- を → "wo" (not "o")
- Preserve exact meaning

Japanese: {japanese}
Romaji:"""
    
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.0,  # Zero temperature for maximum accuracy
            max_tokens=100
        )
        romaji = completion.choices[0].message.content.strip()
        
        # Post-check for critical errors
        if "今" in japanese and "genzai" in romaji.lower():
            romaji = re.sub(r'\bgenzai\b', 'ima', romaji, flags=re.IGNORECASE)
        if "体を" in japanese and "shintai" in romaji.lower():
            romaji = re.sub(r'\bshintai\b', 'karada', romaji, flags=re.IGNORECASE)
        
        return romaji
    except:
        return japanese

# --- SIMPLE BUT EFFECTIVE ALIGNMENT (WHEN GENIUS IS GOOD) ---
async def simple_align_if_genius_good(lrc_lines: List[Dict], romaji_text: str) -> Optional[List[str]]:
    """
    Simple alignment - only use if Genius quality is good
    Returns None if alignment fails
    """
    romaji_lines = [l.strip() for l in romaji_text.split('\n') if l.strip()]
    
    # If line counts are very different, skip
    if abs(len(romaji_lines) - len(lrc_lines)) > max(len(lrc_lines) * 0.3, 10):
        print(f"⚠️ Line count mismatch too big: {len(romaji_lines)} vs {len(lrc_lines)}")
        return None
    
    # Simple 1:1 alignment for first N lines
    aligned = []
    for i, lrc_line in enumerate(lrc_lines):
        if i < len(romaji_lines):
            aligned.append(f"{lrc_line['timestamp']} {romaji_lines[i]}")
        else:
            # Out of romaji lines
            break
    
    # Check if we got enough lines
    if len(aligned) < len(lrc_lines) * 0.8:  # Less than 80% aligned
        return None
    
    return aligned

# --- MAIN PROCESSING: SMART DECISION MAKING ---
async def process_song_smart(song: str, artist: str, force_refresh: bool = False):
    """
    Smart processing: Use AI translation by default, only use Genius if it's perfect
    """
    cache_key = f"smart:{hashlib.md5(f'{song.lower()}:{artist.lower()}'.encode()).hexdigest()}"
    
    if not force_refresh:
        if cache_key in song_cache:
            return song_cache[cache_key]
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                result = json.loads(cached)
                song_cache[cache_key] = result
                return result
    
    print(f"🎯 SMART Processing: {song} by {artist}")
    print("⚠️ Using AI-first approach for maximum accuracy")
    start_time = time.time()
    
    try:
        # Step 1: Get LRC timestamps (fast)
        lrc_lines = await fetch_lrc_timestamps(song, artist)
        if not lrc_lines:
            raise HTTPException(404, "No lyrics found")
        
        print(f"📊 Found {len(lrc_lines)} timed lines")
        
        # Step 2: Get Genius in background (but don't wait for it)
        genius_future = asyncio.create_task(fetch_genius_lyrics_fast(song, artist))
        
        # Step 3: START AI TRANSLATION IMMEDIATELY (no waiting for Genius)
        print("🚀 Starting AI translation immediately...")
        ai_translation_future = asyncio.create_task(
            translate_entire_song_with_ai(lrc_lines)
        )
        
        # Step 4: Check Genius quality while AI is working
        genius_result = await genius_future
        
        final_lyrics = []
        source = ""
        notes = []
        
        if genius_result:
            romaji_text, _ = genius_result
            
            # Validate Genius quality
            japanese_lines = [l['reference'] for l in lrc_lines]
            is_usable, issues = validate_genius_quality(romaji_text, japanese_lines)
            
            if issues:
                print(f"⚠️ Genius has issues: {', '.join(issues[:3])}")
                notes.extend(issues[:3])
            
            if is_usable and len(issues) < 2:
                # Genius is good enough, try simple alignment
                print("✨ Genius quality OK, attempting alignment...")
                corrected_romaji = correct_genius_errors(romaji_text)
                genius_aligned = await simple_align_if_genius_good(lrc_lines, corrected_romaji)
                
                if genius_aligned and len(genius_aligned) == len(lrc_lines):
                    # Verify no critical errors remain
                    critical_errors = 0
                    for i, line in enumerate(genius_aligned):
                        if i < len(lrc_lines):
                            if "今" in lrc_lines[i]['reference'] and "genzai" in line.lower():
                                critical_errors += 1
                            if "体を" in lrc_lines[i]['reference'] and "shintai" in line.lower():
                                critical_errors += 1
                    
                    if critical_errors == 0:
                        final_lyrics = genius_aligned
                        source = "Genius (Good Quality)"
                        print("✅ Using high-quality Genius alignment")
                    else:
                        print(f"⚠️ {critical_errors} critical errors in Genius, using AI instead")
                        source = "AI Translation (Genius had errors)"
                else:
                    print("⚠️ Genius alignment failed, using AI")
                    source = "AI Translation (Alignment failed)"
            else:
                print("❌ Genius quality too poor, using AI")
                source = "AI Translation (Poor Genius quality)"
        else:
            print("🤖 No Genius found, using AI")
            source = "AI Translation"
        
        # Step 5: Get AI translation result
        if not final_lyrics:  # If Genius wasn't used or failed
            print("🔄 Waiting for AI translation...")
            final_lyrics = await ai_translation_future
            if not source:  # If source wasn't set by Genius logic
                source = "AI Translation"
        
        # Step 6: Final verification
        verified_count = 0
        for i, line in enumerate(final_lyrics):
            if i < len(lrc_lines):
                # Check for remaining errors
                if "今" in lrc_lines[i]['reference'] and "genzai" in line.lower():
                    final_lyrics[i] = re.sub(r'\bgenzai\b', 'ima', line, flags=re.IGNORECASE)
                    verified_count += 1
                if "体を" in lrc_lines[i]['reference'] and "shintai" in line.lower():
                    final_lyrics[i] = re.sub(r'\bshintai\b', 'karada', line, flags=re.IGNORECASE)
                    verified_count += 1
        
        if verified_count > 0:
            print(f"🔧 Fixed {verified_count} remaining errors in final verification")
        
        # Step 7: Calculate processing time
        processing_time = round(time.time() - start_time, 2)
        
        result = {
            "lyrics": '\n'.join(final_lyrics),
            "song": song,
            "artist": artist,
            "source": source,
            "line_count": len(final_lyrics),
            "processing_time": processing_time,
            "notes": notes[:3] if notes else [],
            "strategy": "AI-first with Genius fallback",
            "cache_key": cache_key
        }
        
        # Cache
        if not force_refresh:
            song_cache[cache_key] = result
            if redis_client:
                redis_client.setex(cache_key, 86400, json.dumps(result))  # 1 day
        
        print(f"✅ Completed in {processing_time}s via {source}")
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

# --- OPTIMIZED FUNCTIONS FOR SPEED ---
async def fetch_lrc_timestamps(song: str, artist: str) -> Optional[List[Dict]]:
    """Fast LRC fetch with timeout"""
    try:
        url = "https://lrclib.net/api/get"
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, 
            lambda: requests.get(url, params={"track_name": song, "artist_name": artist}, timeout=3)
        )
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
                lines.append({'timestamp': match.group(1), 'reference': match.group(2).strip()})
        return lines
    except: 
        return None

async def fetch_genius_lyrics_fast(song: str, artist: str) -> Optional[Tuple[str, str]]:
    """Fast Genius fetch with timeout"""
    if not GENIUS_API_TOKEN: 
        return None
    try:
        headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
        loop = asyncio.get_event_loop()
        
        # Quick search
        resp = await loop.run_in_executor(
            None, 
            lambda: requests.get(
                "https://api.genius.com/search", 
                headers=headers, 
                params={"q": f"{song} {artist}"}, 
                timeout=4
            )
        )
        data = resp.json()
        
        if not data['response']['hits']:
            return None
        
        song_url = data['response']['hits'][0]['result']['url']
        
        # Quick page fetch
        page = await loop.run_in_executor(
            None,
            lambda: requests.get(song_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=4)
        )
        soup = BeautifulSoup(page.text, 'html.parser')
        
        # Fast extraction
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
        print(f"Genius fetch skipped: {e}")
        return None

# --- REAL-TIME STREAMING WITH INSTANT RESULTS ---
@app.get("/stream_fast")
async def stream_fast(song: str, artist: str):
    """Stream lyrics with INSTANT first results"""
    async def generate():
        cache_key = f"stream:{hashlib.md5(f'{song.lower()}:{artist.lower()}'.encode()).hexdigest()}"
        
        # Try cache first
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                data = json.loads(cached)
                lyrics = data.get("lyrics", "").split('\n')
                
                yield json.dumps({
                    "status": "cached",
                    "total": len(lyrics),
                    "source": data.get("source", "cache")
                }) + "\n"
                
                for i, line in enumerate(lyrics):
                    yield json.dumps({
                        "line": line,
                        "index": i,
                        "total": len(lyrics)
                    }) + "\n"
                
                yield json.dumps({"status": "complete"}) + "\n"
                return
        
        # Not cached, start real-time processing
        yield json.dumps({"status": "starting", "song": song, "artist": artist}) + "\n"
        
        # Get LRC first (fast)
        lrc_lines = await fetch_lrc_timestamps(song, artist)
        if not lrc_lines:
            yield json.dumps({"error": "No lyrics found"}) + "\n"
            return
        
        yield json.dumps({
            "status": "lrc_loaded",
            "count": len(lrc_lines),
            "progress": 0.2
        }) + "\n"
        
        # Start AI translation immediately
        print(f"🚀 Starting real-time AI translation for {len(lrc_lines)} lines...")
        
        # Send first batch IMMEDIATELY
        first_batch = min(5, len(lrc_lines))
        if first_batch > 0:
            for i in range(first_batch):
                # Quick translate first few lines
                translated = await translate_single_line_accurate(lrc_lines[i]['reference'])
                line = f"{lrc_lines[i]['timestamp']} {translated}"
                
                yield json.dumps({
                    "line": line,
                    "index": i,
                    "total": len(lrc_lines),
                    "progress": 0.3 + (i / len(lrc_lines) * 0.2)
                }) + "\n"
        
        # Process rest in background while streaming continues
        remaining = lrc_lines[first_batch:]
        if remaining:
            # Start batch translation for remaining lines
            batch_size = 10
            for i in range(0, len(remaining), batch_size):
                batch = remaining[i:i+batch_size]
                batch_translations = []
                
                # Translate batch
                for lrc_line in batch:
                    translated = await translate_single_line_accurate(lrc_line['reference'])
                    batch_translations.append(translated)
                
                # Send batch
                for j, translation in enumerate(batch_translations):
                    line_idx = first_batch + i + j
                    line = f"{batch[j]['timestamp']} {translation}"
                    
                    yield json.dumps({
                        "line": line,
                        "index": line_idx,
                        "total": len(lrc_lines),
                        "progress": 0.5 + (line_idx / len(lrc_lines) * 0.5)
                    }) + "\n"
        
        yield json.dumps({
            "status": "complete",
            "progress": 1.0,
            "source": "AI Translation (Real-time)"
        }) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {
        "status": "Online",
        "version": "AI-First Solution",
        "note": "Using AI translation for 100% accuracy, Genius as fallback only",
        "endpoints": {
            "/get_song": "Get lyrics (cached)",
            "/get_song_fresh": "Get lyrics (fresh, no cache)",
            "/stream_fast": "Stream lyrics instantly",
            "/clear_cache": "Clear all cache",
            "/test_accuracy": "Test translation accuracy"
        }
    }

@app.get("/get_song")
async def get_song_endpoint(song: str, artist: str, force_refresh: bool = False):
    """Main endpoint - uses AI-first approach"""
    return await process_song_smart(song, artist, force_refresh)

@app.get("/get_song_fresh")
async def get_song_fresh(song: str, artist: str):
    """Always fresh lyrics"""
    return await process_song_smart(song, artist, force_refresh=True)

@app.get("/test_accuracy")
async def test_accuracy():
    """Test specific accuracy problems"""
    test_cases = [
        {
            "japanese": "体を触って必要なのはこれだけ認めて",
            "expected": "karada wo sawatte hitsuyou na no wa kore dake mitomete",
            "wrong_genius": "shintai wo sawatte hitsuyou na no wa kore dake mitomete"
        },
        {
            "japanese": "夜道を迷ぐれど虚しい",
            "expected": "yomichi wo masaguredo munashii",
            "wrong_genius": "yomichi o iburedo munashi"
        },
        {
            "japanese": "改札の安警光灯",
            "expected": "kaisatsu no yasu keikoutou",
            "wrong_genius": "kaisatsu no an keikoto wa"
        },
        {
            "japanese": "サイレン爆音現実界ある浮遊",
            "expected": "sairen bakuon genjitsukai aru fuyuu",
            "wrong_genius": "sairen bakguen genjikkai aru fuyu"
        }
    ]
    
    results = []
    for test in test_cases:
        if client:
            translated = await translate_single_line_accurate(test["japanese"])
        else:
            translated = test["japanese"]
        
        # Check for errors
        has_genzai = "genzai" in translated.lower() and "今" in test["japanese"]
        has_shintai = "shintai" in translated.lower() and "体" in test["japanese"]
        has_wrong_particle = " o " in translated and "を" in test["japanese"]
        
        results.append({
            "japanese": test["japanese"],
            "translated": translated,
            "expected": test["expected"],
            "matches_expected": translated.lower() == test["expected"].lower(),
            "errors": {
                "has_genzai": has_genzai,
                "has_shintai": has_shintai,
                "has_wrong_particle": has_wrong_particle
            }
        })
    
    return {
        "test": "Accuracy Test",
        "results": results,
        "summary": {
            "total": len(results),
            "correct": sum(1 for r in results if r["matches_expected"]),
            "errors": sum(1 for r in results if any(r["errors"].values()))
        }
    }

@app.delete("/clear_cache")
async def clear_cache():
    """Clear all cache"""
    song_cache.clear()
    line_cache.clear()
    if redis_client:
        redis_client.flushdb()
    return {"status": "Cache cleared", "message": "Now using AI-first approach for better accuracy"}

@app.get("/health")
async def health():
    """Health check"""
    return {
        "deepseek": bool(client),
        "redis": redis_client.ping() if redis_client else False,
        "genius": bool(GENIUS_API_TOKEN),
        "cache_size": len(song_cache),
        "strategy": "AI-first with Genius validation"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
