from fastapi import FastAPI, HTTPException, BackgroundTasks
from openai import AsyncOpenAI  # <--- Standard client for DeepSeek
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
# Get this from: https://platform.deepseek.com/
API_KEY = os.environ.get("DEEPSEEK_API_KEY") 
REDIS_URL = os.environ.get("REDIS_URL")

# DEEPSEEK SETTINGS
# 'deepseek-chat' is V3 (Fast & Smart). 'deepseek-reasoner' is R1 (Slower, thinks hard).
# For lyrics, 'deepseek-chat' is perfect.
MODEL_NAME = "deepseek-chat"
DAILY_REQUEST_LIMIT = 5000  # DeepSeek is cheap, so we can allow way more usage!

client = None
redis_client = None

# --- 1. SETUP SYSTEMS ---
def setup_systems():
    global client, redis_client
    
    # Setup DeepSeek (via OpenAI Client)
    if not API_KEY:
        print("❌ CRITICAL: DEEPSEEK_API_KEY missing!")
    else:
        try:
            # DeepSeek uses the OpenAI SDK but with a different Base URL
            client = AsyncOpenAI(
                api_key=API_KEY, 
                base_url="https://api.deepseek.com"
            )
            print(f"✅ DeepSeek AI Online: {MODEL_NAME}")
        except Exception as e:
            print(f"❌ AI Error: {e}")

    # Setup Redis
    if not REDIS_URL:
        print("⚠️ WARNING: Redis missing! Cache will not survive restarts.")
    else:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            redis_client.ping()
            print("✅ Redis Online: Persistent Cache Active")
        except Exception as e:
            print(f"❌ Redis Error: {e}")

setup_systems()

# --- 2. QUOTA MANAGEMENT (Redis-Backed) ---
def check_quota() -> bool:
    """Tracks usage so you don't burn money accidentally"""
    if not redis_client: return True # Fallback if Redis dies
    
    today = datetime.now().strftime("%Y-%m-%d")
    quota_key = f"quota:deepseek:{today}"
    
    try:
        current = int(redis_client.get(quota_key) or 0)
        if current >= DAILY_REQUEST_LIMIT:
            print(f"⛔ Daily Limit Hit: {current}/{DAILY_REQUEST_LIMIT}")
            return False
            
        pipe = redis_client.pipeline()
        pipe.incr(quota_key)
        pipe.expire(quota_key, 86400) # Reset daily
        pipe.execute()
        return True
    except: return True

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
    except: pass

# --- 4. AI CONVERSION LOGIC ---
async def convert_batch(texts: List[str]) -> List[str]:
    if not texts or not client: return texts
    
    # 1. Check Caches
    uncached_idxs = []
    results = [None] * len(texts)
    
    for i, text in enumerate(texts):
        if text in line_cache:
            results[i] = line_cache[text]
            continue
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        redis_key = f"line:{text_hash}"
        cached = get_from_redis(redis_key)
        
        if cached:
            results[i] = cached['romaji']
            line_cache[text] = cached['romaji']
        else:
            uncached_idxs.append(i)
            
    if not uncached_idxs: return results

    # 2. Check Quota
    if not check_quota():
        for idx in uncached_idxs: results[idx] = texts[idx]
        return results

    # 3. Call DeepSeek
    subset = [texts[i] for i in uncached_idxs]
    
    prompt = f"""Convert these {len(subset)} Japanese lyric lines to Romaji (Hepburn).
    Output EXACTLY {len(subset)} lines. Format: "1. romaji"
    NO intro/outro/thinking.
    
    INPUT:
    {chr(10).join([f"{i+1}. {t}" for i, t in enumerate(subset)])}"""
    
    try:
        completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise Japanese-to-Romaji converter."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL_NAME,
            temperature=0.1,
            max_tokens=4096,
            stream=False
        )
        
        content = completion.choices[0].message.content.strip()
        lines = content.split('\n')
        
        for line in lines:
            match = re.match(r'^(\d+)[\.\)]\s*(.+)', line.strip())
            if match:
                idx_in_subset = int(match.group(1)) - 1
                if 0 <= idx_in_subset < len(uncached_idxs):
                    original_idx = uncached_idxs[idx_in_subset]
                    romaji = match.group(2).strip()
                    
                    results[original_idx] = romaji
                    line_cache[texts[original_idx]] = romaji
                    
                    thash = hashlib.md5(texts[original_idx].encode()).hexdigest()
                    save_to_redis(f"line:{thash}", {'romaji': romaji})
                    
    except Exception as e:
        print(f"❌ DeepSeek Error: {e}")

    # Fill failures
    for i in range(len(results)):
        if results[i] is None: results[i] = texts[i]
        
    return results

# --- 5. SONG PROCESSING ---
async def process_song(song: str, artist: str):
    cache_key = f"song:{song.lower()}:{artist.lower()}"
    
    if cache_key in song_cache: return song_cache[cache_key]
    cached = get_from_redis(cache_key)
    if cached: 
        song_cache[cache_key] = cached
        return cached

    # Fetch Lyrics
    try:
        url = "https://lrclib.net/api/get"
        resp = requests.get(url, params={"track_name": song, "artist_name": artist}, timeout=10)
        data = resp.json()
        raw = data.get("syncedLyrics") or data.get("plainLyrics")
        if not raw: return None
    except: return None

    # Parse
    lines = []
    for l in raw.split('\n'):
        if not l.strip(): continue
        m = re.match(r'(\[\d+:\d+\.\d+\])\s*(.*)', l)
        if m: lines.append({'time': m.group(1), 'text': m.group(2).strip()})
        else: lines.append({'time': '', 'text': l.strip()})

    # Batch Convert (DeepSeek handles larger contexts well)
    texts = [x['text'] for x in lines if x['text']]
    romaji_map = {}
    
    # Process in batches of 40 (DeepSeek has 64k context, it can handle it)
    for i in range(0, len(texts), 40):
        batch = texts[i:i+40]
        converted = await convert_batch(batch)
        for orig, conv in zip(batch, converted):
            romaji_map[orig] = conv
            
    final_lines = []
    for l in lines:
        romaji = romaji_map.get(l['text'], l['text'])
        final_lines.append(f"{l['time']} {romaji}")

    result = {
        "lyrics": '\n'.join(final_lines), 
        "song": song, 
        "artist": artist,
        "cached": False,
        "model": MODEL_NAME
    }
    
    song_cache[cache_key] = result
    save_to_redis(cache_key, result)
    return result

# --- 6. ENDPOINTS ---
@app.get("/")
async def root():
    return {"status": "Online", "provider": "DeepSeek", "model": MODEL_NAME}

@app.get("/convert")
async def convert_single_line(text: str = ""):
    if not text: raise HTTPException(400, "No text provided")
    results = await convert_batch([text])
    return {"original": text, "romaji": results[0]}

@app.get("/get_song")
async def get_song_endpoint(song: str, artist: str):
    res = await process_song(song, artist)
    if res: return res
    raise HTTPException(404, "Song not found")

@app.get("/cache_status")
async def status():
    used = 0
    if redis_client:
        today = datetime.now().strftime("%Y-%m-%d")
        used = redis_client.get(f"quota:deepseek:{today}") or 0
    return {
        "redis": redis_client is not None,
        "quota_used": int(used),
        "quota_limit": DAILY_REQUEST_LIMIT,
        "model": MODEL_NAME
    }
