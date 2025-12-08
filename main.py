from fastapi import FastAPI, HTTPException, BackgroundTasks
import os
import re
import asyncio
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import requests

# --- NEW IMPORT ---
from groq import AsyncGroq

app = FastAPI()

# --- 1. SETUP AI WITH RATE LIMITING ---
# CHANGE: Get Groq Key instead of Gemini
API_KEY = os.environ.get("GROQ_API_KEY") 
REDIS_URL = os.environ.get("REDIS_URL")

client = None # Replaces 'model' object
GROQ_MODEL = "llama-3.3-70b-versatile" # Very fast, high intelligence

redis_client = None

# TRACKING (Groq has much higher limits, but tracking is still good practice)
daily_request_count = 0
# Groq's free tier is generous (approx 30 requests/minute), so we can raise this safe limit
daily_request_limit = 1000 
last_reset_date = datetime.now().date()

def check_and_increment_quota() -> bool:
    """Check if we can make an API request without exceeding quota"""
    global daily_request_count, last_reset_date
    
    # Reset counter if it's a new day
    today = datetime.now().date()
    if today > last_reset_date:
        daily_request_count = 0
        last_reset_date = today
        print(f"📅 Daily quota reset: {daily_request_count}/{daily_request_limit}")
    
    # Check if we're under the limit
    if daily_request_count >= daily_request_limit:
        print(f"⛔ QUOTA EXCEEDED: {daily_request_count}/{daily_request_limit} requests used today")
        return False
    
    daily_request_count += 1
    print(f"📊 API Usage: {daily_request_count}/{daily_request_limit} requests today")
    return True

def setup_ai():
    global client
    if not API_KEY:
        print("❌ CRITICAL ERROR: GROQ_API_KEY is missing!")
        return

    try:
        # CHANGE: Initialize Groq Client
        client = AsyncGroq(api_key=API_KEY)
        print(f"✅ SUCCESS: Groq Client initialized using model '{GROQ_MODEL}'")
        
    except Exception as e:
        print(f"❌ Error setting up Groq: {e}")

def setup_redis():
    global redis_client
    if not REDIS_URL:
        print("⚠️ ⚠️ ⚠️ REDIS_URL NOT SET - CACHE WILL BE LOST ON RESTART! ⚠️ ⚠️ ⚠️")
        return
    
    try:
        import redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        print("✅ Redis connected successfully!")
    except ImportError:
        print("⚠️ Redis library not installed. Run: pip install redis")
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")

setup_ai()
setup_redis()

# --- 2. PERSISTENT STORAGE ---
line_cache = {}
song_cache = {}

def get_from_persistent(key: str, prefix: str = "song") -> Optional[dict]:
    """Get from Redis"""
    if redis_client:
        try:
            data = redis_client.get(f"{prefix}:{key}")
            return json.loads(data) if data else None
        except Exception as e:
            print(f"Redis GET error: {e}")
    return None

def save_to_persistent(key: str, data: dict, prefix: str = "song", ttl: int = 2592000):
    """Save to Redis (30 days)"""
    if redis_client:
        try:
            redis_client.setex(f"{prefix}:{key}", ttl, json.dumps(data))
            return True
        except Exception as e:
            print(f"Redis SET error: {e}")
    return False

def get_song_cached(song: str, artist: str) -> Optional[dict]:
    """Get song from memory OR Redis"""
    cache_key = f"{song.lower()}::{artist.lower()}"
    
    if cache_key in song_cache:
        return song_cache[cache_key]
    
    persistent_data = get_from_persistent(cache_key)
    if persistent_data:
        song_cache[cache_key] = persistent_data
        return persistent_data
    
    return None

def save_song_cached(song: str, artist: str, data: dict):
    """Save song to both memory AND Redis"""
    cache_key = f"{song.lower()}::{artist.lower()}"
    song_cache[cache_key] = data
    save_to_persistent(cache_key, data)

processing_queue = set()

# --- 3. SMART AI CONVERSION (GROQ VERSION) ---
async def convert_batch_smart(texts: List[str]) -> List[str]:
    """Convert with quota protection and caching"""
    if not texts or not client:
        return texts
    
    # Check ALL caches first
    uncached_indices = []
    results = [None] * len(texts)
    
    for i, text in enumerate(texts):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Try memory cache
        if text in line_cache:
            results[i] = line_cache[text]
            continue
        
        # Try Redis cache
        cached_romaji = get_from_persistent(text_hash, prefix="line")
        if cached_romaji:
            romaji_text = cached_romaji.get('romaji', text)
            results[i] = romaji_text
            line_cache[text] = romaji_text
        else:
            uncached_indices.append(i)
    
    if not uncached_indices:
        print(f"✅ All {len(texts)} lines from cache!")
        return results
    
    # Check quota BEFORE making API call
    if not check_and_increment_quota():
        print(f"⛔ QUOTA EXCEEDED - Returning original text for {len(uncached_indices)} lines")
        for idx in uncached_indices:
            results[idx] = texts[idx]
        return results
    
    # Make ONE API call for remaining lines
    uncached_texts = [texts[i] for i in uncached_indices]
    
    # CHANGE: Updated Prompt structure for Llama 3 models
    system_prompt = "You are a Japanese to Romaji converter. You output strictly formatted numbered lists."
    user_prompt = f"""Convert these {len(uncached_texts)} Japanese song lyrics to Romaji (Hepburn romanization).

RULES:
1. Output EXACTLY {len(uncached_texts)} lines.
2. Format strictly as: "1. romaji text"
3. Do not add intro or outro text.

INPUT LINES:
{chr(10).join([f"{i+1}. {text}" for i, text in enumerate(uncached_texts)])}"""
    
    try:
        # CHANGE: Groq API Call Structure
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=GROQ_MODEL,
            temperature=0.1, # Keep it precise
            max_tokens=4096,
        )

        # CHANGE: How we get text from response
        response_text = chat_completion.choices[0].message.content
        converted_lines = response_text.strip().split('\n')
        
        for line in converted_lines:
            line = line.strip()
            if not line:
                continue
            
            match = re.match(r'^(\d+)[\.\):\-\s]+(.+)', line)
            if match:
                line_num = int(match.group(1))
                romaji = match.group(2).strip()
                
                if 1 <= line_num <= len(uncached_texts):
                    idx = uncached_indices[line_num - 1]
                    results[idx] = romaji
                    original = texts[idx]
                    line_cache[original] = romaji
                    
                    # Save to Redis
                    text_hash = hashlib.md5(original.encode()).hexdigest()
                    save_to_persistent(text_hash, {'romaji': romaji}, prefix="line")
        
        print(f"✅ Converted {len(uncached_texts)} new lines via Groq")
        
    except Exception as e:
        print(f"❌ Conversion error: {e}")
    
    # Fill remaining with original text
    for i in range(len(results)):
        if results[i] is None:
            results[i] = texts[i]
    
    return results

# --- 4. HELPER FUNCTIONS ---
def extract_lyrics_lines(raw_lyrics: str) -> List[dict]:
    """Extract lines with timestamps"""
    lines = []
    for line in raw_lyrics.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        match = re.match(r'(\[\d+:\d+\.\d+\])\s*(.*)', line)
        if match:
            timestamp, text = match.groups()
            if text.strip():
                lines.append({'timestamp': timestamp, 'text': text.strip()})
        elif line:
            lines.append({'timestamp': '', 'text': line})
    
    return lines

async def fetch_lyrics_async(song: str, artist: str) -> Optional[str]:
    """Fetch lyrics"""
    try:
        url = "https://lrclib.net/api/get"
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: requests.get(
                url, 
                params={"track_name": song, "artist_name": artist}, 
                timeout=15
            )
        )
        
        if resp.status_code != 200:
            return None
            
        data = resp.json()
        return data.get("syncedLyrics") or data.get("plainLyrics")
    except Exception as e:
        print(f"Lyrics fetch error: {e}")
        return None

# --- 5. SONG PREPARATION ---
async def prepare_song_internal(song: str, artist: str) -> Optional[dict]:
    """Fetch and convert song"""
    cache_key = f"{song.lower()}::{artist.lower()}"
    
    # Check cache FIRST
    cached = get_song_cached(song, artist)
    if cached:
        print(f"✓ Cache hit: {song} - {artist}")
        return cached
    
    if cache_key in processing_queue:
        print(f"⏳ Already processing: {song} - {artist}")
        for _ in range(30):
            await asyncio.sleep(1)
            cached = get_song_cached(song, artist)
            if cached:
                return cached
        return None
    
    # Check if we have quota left
    if daily_request_count >= daily_request_limit - 2: 
        print(f"⛔ QUOTA TOO LOW - Cannot convert {song}.")
        return None
    
    processing_queue.add(cache_key)
    print(f"🎵 Starting conversion: {song} - {artist}")
    
    try:
        raw_lyrics = await fetch_lyrics_async(song, artist)
        
        if not raw_lyrics:
            print(f"❌ Lyrics not found: {song} - {artist}")
            processing_queue.discard(cache_key)
            return None
        
        lyrics_lines = extract_lyrics_lines(raw_lyrics)
        if not lyrics_lines:
            print(f"❌ Empty lyrics: {song} - {artist}")
            processing_queue.discard(cache_key)
            return None
        
        texts_to_convert = [line['text'] for line in lyrics_lines]
        
        # Groq is fast, we can handle batches of 30 easily
        BATCH_SIZE = 30
        all_romaji = []
        
        for i in range(0, len(texts_to_convert), BATCH_SIZE):
            batch = texts_to_convert[i:i + BATCH_SIZE]
            romaji_batch = await convert_batch_smart(batch)
            all_romaji.extend(romaji_batch)
            
            # Tiny sleep just to be safe, though Groq is very fast
            if i + BATCH_SIZE < len(texts_to_convert):
                await asyncio.sleep(0.2)
        
        result_lyrics = []
        for i, line_data in enumerate(lyrics_lines):
            if line_data['timestamp']:
                result_lyrics.append(f"{line_data['timestamp']} {all_romaji[i]}")
            else:
                result_lyrics.append(all_romaji[i])
        
        result = {
            "lyrics": '\n'.join(result_lyrics),
            "total_lines": len(lyrics_lines),
            "cached_at": datetime.now().isoformat(),
            "song": song,
            "artist": artist
        }
        
        save_song_cached(song, artist, result)
        print(f"✅ Conversion complete: {song} - {artist} ({len(lyrics_lines)} lines)")
        
        processing_queue.discard(cache_key)
        return result
        
    except Exception as e:
        print(f"❌ Error: {song} - {artist}: {e}")
        processing_queue.discard(cache_key)
        return None

@app.on_event("startup")
async def startup_event():
    print("✅ Server started with GROQ AI")
    print(f"📊 Daily limit set to: {daily_request_limit} requests")
    if not redis_client:
        print("⚠️⚠️⚠️ WARNING: No Redis - cache will reset on restart! ⚠️⚠️⚠️")

# --- 6. API ENDPOINTS ---

@app.get("/")
async def root():
    return {
        "status": "Romaji Converter API Running (Groq Powered)",
        "model": GROQ_MODEL,
        "quota": {
            "used_today": daily_request_count,
            "daily_limit": daily_request_limit,
            "remaining": daily_request_limit - daily_request_count
        },
        "cache": {
            "songs": len(song_cache),
            "lines": len(line_cache),
            "redis_connected": redis_client is not None
        }
    }

@app.get("/convert")
async def convert_romaji(text: str = ""):
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    if text in line_cache:
        return {"original": text, "romaji": line_cache[text], "cached": True}

    if not client:
        return {"original": text, "romaji": text, "cached": False}
    
    if daily_request_count >= daily_request_limit:
        return {"original": text, "romaji": text, "error": "Quota exceeded"}

    try:
        results = await convert_batch_smart([text])
        return {"original": text, "romaji": results[0], "cached": False}
    except Exception as e:
        print(f"Conversion error: {e}")
        return {"original": text, "romaji": text, "cached": False}

@app.get("/get_song")
async def get_song(song: str, artist: str):
    result = await prepare_song_internal(song, artist)
    
    if result:
        return {
            "status": "ready",
            "quota_used": daily_request_count,
            "quota_remaining": daily_request_limit - daily_request_count,
            **result
        }
    else:
        raise HTTPException(
            status_code=404, 
            detail="Song not found, conversion failed, or quota exceeded"
        )
