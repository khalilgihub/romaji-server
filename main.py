from fastapi import FastAPI, HTTPException, BackgroundTasks
from groq import AsyncGroq
import requests
import os
import re
import asyncio
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
import json
import redis

app = FastAPI()

# --- CONFIGURATION ---
API_KEY = os.environ.get("GROQ_API_KEY")
REDIS_URL = os.environ.get("REDIS_URL")

# OPTIMIZATION: Use the 8B model. It has 5x higher limits (500k tokens/day) vs 70B (100k).
GROQ_MODEL = "llama-3.1-8b-instant" 
DAILY_REQUEST_LIMIT = 300  # Safe limit for 8B model (approx 500k tokens)

client = None
redis_client = None

# --- 1. SETUP ---
def setup_systems():
    global client, redis_client
    
    # Setup Groq
    if not API_KEY:
        print("❌ CRITICAL: GROQ_API_KEY missing!")
    else:
        try:
            client = AsyncGroq(api_key=API_KEY)
            print(f"✅ AI Online: {GROQ_MODEL}")
        except Exception as e:
            print(f"❌ AI Error: {e}")

    # Setup Redis
    if not REDIS_URL:
        print("⚠️ WARNING: Redis missing! Quotas will reset on restart.")
    else:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            redis_client.ping()
            print("✅ Redis Online: Cache & Quota synced")
        except Exception as e:
            print(f"❌ Redis Error: {e}")

setup_systems()

# --- 2. SMART QUOTA MANAGEMENT (Redis-Backed) ---
def check_quota() -> bool:
    """Check quota using Redis so it survives server restarts"""
    if not redis_client:
        # Fallback to in-memory if Redis is down (risky but necessary fallback)
        return True 

    today = datetime.now().strftime("%Y-%m-%d")
    quota_key = f"quota:{today}"
    
    try:
        # Get current usage
        current = int(redis_client.get(quota_key) or 0)
        
        if current >= DAILY_REQUEST_LIMIT:
            print(f"⛔ Daily Limit Hit: {current}/{DAILY_REQUEST_LIMIT}")
            return False
            
        # Increment usage (expires in 24 hours)
        pipe = redis_client.pipeline()
        pipe.incr(quota_key)
        pipe.expire(quota_key, 86400) # Auto-delete next day
        pipe.execute()
        
        print(f"📊 Daily Usage: {current + 1}/{DAILY_REQUEST_LIMIT}")
        return True
    except Exception as e:
        print(f"⚠️ Quota Check Error: {e}")
        return True # Default to allow if Redis fails

# --- 3. STORAGE & CACHING ---
line_cache = {}
song_cache = {}

def get_from_redis(key: str):
    if not redis_client: return None
    try:
        data = redis_client.get(key)
        return json.loads(data) if data else None
    except: return None

def save_to_redis(key: str, data: dict, ttl=2592000): # 30 days
    if not redis_client: return
    try:
        redis_client.setex(key, ttl, json.dumps(data))
    except Exception as e: print(f"Redis Save Error: {e}")

# --- 4. AI CONVERSION LOGIC ---
async def convert_batch(texts: List[str]) -> List[str]:
    if not texts or not client: return texts
    
    # 1. Check Caches (Memory -> Redis)
    uncached_idxs = []
    results = [None] * len(texts)
    
    for i, text in enumerate(texts):
        # Memory
        if text in line_cache:
            results[i] = line_cache[text]
            continue
            
        # Redis
        text_hash = hashlib.md5(text.encode()).hexdigest()
        redis_key = f"line:{text_hash}"
        cached = get_from_redis(redis_key)
        if cached:
            results[i] = cached['romaji']
            line_cache[text] = cached['romaji'] # Refill memory
        else:
            uncached_idxs.append(i)
            
    if not uncached_idxs: return results

    # 2. Check Quota BEFORE calling AI
    if not check_quota():
        print("⚠️ Quota exceeded, skipping AI conversion")
        # Return original text for remaining lines
        for idx in uncached_idxs: results[idx] = texts[idx]
        return results

    # 3. Call Groq AI
    subset = [texts[i] for i in uncached_idxs]
    
    prompt = f"""Convert these {len(subset)} Japanese lyrics to Romaji (Hepburn).
    Output EXACTLY {len(subset)} lines. Format: "1. romaji"
    NO intro/outro.
    
    INPUT:
    {chr(10).join([f"{i+1}. {t}" for i, t in enumerate(subset)])}"""
    
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=4096
        )
        
        lines = completion.choices[0].message.content.strip().split('\n')
        
        # Parse output
        for line in lines:
            match = re.match(r'^(\d+)[\.\)]\s*(.+)', line.strip())
            if match:
                idx_in_subset = int(match.group(1)) - 1
                if 0 <= idx_in_subset < len(uncached_idxs):
                    original_idx = uncached_idxs[idx_in_subset]
                    romaji = match.group(2).strip()
                    
                    # Save results
                    results[original_idx] = romaji
                    line_cache[texts[original_idx]] = romaji
                    
                    # Persist to Redis
                    thash = hashlib.md5(texts[original_idx].encode()).hexdigest()
                    save_to_redis(f"line:{thash}", {'romaji': romaji})
                    
    except Exception as e:
        print(f"❌ AI Error: {e}")

    # Fill failures with original text
    for i in range(len(results)):
        if results[i] is None: results[i] = texts[i]
        
    return results

# --- 5. SONG FETCHING ---
async def process_song(song: str, artist: str):
    cache_key = f"song:{song.lower()}:{artist.lower()}"
    
    # Check Cache
    if cache_key in song_cache: return song_cache[cache_key]
    cached = get_from_redis(cache_key)
    if cached: 
        song_cache[cache_key] = cached
        return cached

    # Fetch from LRCLIB
    print(f"🎵 Fetching: {song}")
    try:
        url = "https://lrclib.net/api/get"
        resp = requests.get(url, params={"track_name": song, "artist_name": artist}, timeout=10)
        data = resp.json()
        raw = data.get("syncedLyrics") or data.get("plainLyrics")
        if not raw: return None
    except: return None

    # Parse Lines
    lines = []
    for l in raw.split('\n'):
        if not l.strip(): continue
        m = re.match(r'(\[\d+:\d+\.\d+\])\s*(.*)', l)
        if m: lines.append({'time': m.group(1), 'text': m.group(2).strip()})
        else: lines.append({'time': '', 'text': l.strip()})

    # Batch Convert
    texts = [x['text'] for x in lines if x['text']]
    
    # Process in chunks of 30
    romaji_map = {}
    for i in range(0, len(texts), 30):
        batch = texts[i:i+30]
        converted = await convert_batch(batch)
        for orig, conv in zip(batch, converted):
            romaji_map[orig] = conv
            
    # Rebuild
    final_lines = []
    for l in lines:
        romaji = romaji_map.get(l['text'], l['text'])
        final_lines.append(f"{l['time']} {romaji}")

    result = {
        "lyrics": '\n'.join(final_lines),
        "song": song, 
        "artist": artist,
        "cached": False
    }
    
    # Save to Cache
    song_cache[cache_key] = result
    save_to_redis(cache_key, result)
    return result

# --- 6. ENDPOINTS ---
@app.get("/")
async def root():
    return {"status": "Online", "model": GROQ_MODEL}

@app.get("/get_song")
async def get_song_endpoint(song: str, artist: str):
    res = await process_song(song, artist)
    if res: return res
    raise HTTPException(404, "Song not found")

@app.get("/cache_status")
async def status():
    # Get real quota from Redis
    used = 0
    if redis_client:
        today = datetime.now().strftime("%Y-%m-%d")
        used = redis_client.get(f"quota:{today}") or 0
        
    return {
        "redis": redis_client is not None,
        "quota_used": int(used),
        "quota_limit": DAILY_REQUEST_LIMIT,
        "model": GROQ_MODEL
    }
