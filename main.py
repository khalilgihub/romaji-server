from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
import requests
import os
import re
import hashlib
from datetime import datetime
from typing import List, Optional, Dict
import json
import redis
from bs4 import BeautifulSoup
import asyncio

app = FastAPI()

# --- CONFIGURATION ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") 
GENIUS_API_TOKEN = os.environ.get("GENIUS_API_TOKEN")
REDIS_URL = os.environ.get("REDIS_URL")

# MODEL SETTING:
# "deepseek-chat" (V3) is BEST for speed and formatting.
# "deepseek-reasoner" (R1) is smarter but slower/more expensive.
DEEPSEEK_MODEL = "deepseek-chat" 

# NO LIMIT SETTING:
# Set to a number you will practically never reach (1 Billion)
DAILY_REQUEST_LIMIT = 1_000_000_000

client = None
redis_client = None
line_cache = {}
song_cache = {}

def setup_systems():
    global client, redis_client
    if DEEPSEEK_API_KEY:
        try:
            client = AsyncOpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com"
            )
            print(f"✅ DeepSeek AI Online: {DEEPSEEK_MODEL} (Unlimited Mode)")
        except Exception as e:
            print(f"❌ AI Error: {e}")
    
    if REDIS_URL:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            redis_client.ping()
            print("✅ Redis Online")
        except Exception as e:
            print(f"❌ Redis Error: {e}")
            
    if GENIUS_API_TOKEN:
        print("✅ Genius API Token Loaded")
    else:
        print("⚠️ Genius API Token Missing")

setup_systems()

# --- GENIUS API FUNCTIONS ---

def search_genius(song: str, artist: str) -> Optional[str]:
    if not GENIUS_API_TOKEN: return None
    try:
        headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
        response = requests.get(
            "https://api.genius.com/search", 
            headers=headers, 
            params={"q": f"{song} {artist}"}, 
            timeout=10
        )
        data = response.json()
        if data['response']['hits']:
            return data['response']['hits'][0]['result']['url']
    except Exception as e:
        print(f"Genius Search Error: {e}")
    return None

def scrape_genius_lyrics(song_url: str) -> Optional[Dict[str, str]]:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(song_url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
        if not lyrics_divs: return None
        
        full_text = []
        for div in lyrics_divs:
            for element in div.descendants:
                if element.name == 'br': full_text.append('\n')
                elif isinstance(element, str):
                    text = element.strip()
                    if text: full_text.append(text)
        
        lyrics = '\n'.join(full_text)
        
        # Simple detection logic
        lines = lyrics.split('\n')
        japanese_lines = []
        romaji_lines = []
        
        for line in lines:
            if not line.strip(): continue
            if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line):
                japanese_lines.append(line)
            elif re.match(r'^[a-zA-Z\s\-\'\.!?,()]+$', line) and len(line) > 3:
                romaji_lines.append(line)
        
        result = {}
        if japanese_lines and romaji_lines and len(romaji_lines) > len(japanese_lines) * 0.5:
            result['japanese'] = '\n'.join(japanese_lines)
            result['romaji'] = '\n'.join(romaji_lines)
            result['source'] = 'genius_dual'
        elif romaji_lines and len(romaji_lines) > 10:
            result['romaji'] = '\n'.join(romaji_lines)
            result['source'] = 'genius_romaji'
        elif japanese_lines:
            result['japanese'] = '\n'.join(japanese_lines)
            result['source'] = 'genius_japanese'
        else:
            result['lyrics'] = lyrics
            result['source'] = 'genius_mixed'
            
        return result
    except Exception as e:
        print(f"Genius Scraping Error: {e}")
        return None

def fetch_from_lrclib(song: str, artist: str, duration: int = 0) -> Optional[Dict]:
    try:
        params = {"track_name": song, "artist_name": artist}
        if duration > 0: params["duration"] = duration
        response = requests.get("https://lrclib.net/api/get", params=params, timeout=10)
        data = response.json()
        synced = data.get("syncedLyrics")
        plain = data.get("plainLyrics")
        if synced or plain:
            return {"synced": synced, "plain": plain, "source": "lrclib"}
    except: pass
    return None

# --- SMART LYRICS FETCHER ---

async def fetch_lyrics_smart(song: str, artist: str, duration: int = 0) -> Optional[Dict]:
    cache_key = f"lyrics:{song.lower()}:{artist.lower()}"
    
    if cache_key in song_cache: return song_cache[cache_key]
    cached = get_from_redis(cache_key)
    if cached: 
        song_cache[cache_key] = cached
        return cached
    
    print(f"🔍 Fetching: {song} - {artist}")
    
    # 1. Genius
    genius_url = search_genius(song, artist)
    if genius_url:
        genius_data = scrape_genius_lyrics(genius_url)
        if genius_data:
            if 'romaji' in genius_data:
                result = {"lyrics": genius_data['romaji'], "type": "romaji", "source": genius_data['source'], "ai_used": False}
                save_and_cache(cache_key, result)
                return result
            if 'japanese' in genius_data:
                print("🤖 Genius found Japanese, converting...")
                romaji = await convert_japanese_to_romaji(genius_data['japanese'])
                result = {"lyrics": romaji, "type": "romaji_converted", "source": "genius_converted", "ai_used": True}
                save_and_cache(cache_key, result)
                return result

    # 2. LRCLib
    lrclib_data = fetch_from_lrclib(song, artist, duration)
    if lrclib_data:
        lyrics_text = lrclib_data['synced'] or lrclib_data['plain']
        lines = parse_lrc_format(lyrics_text)
        
        needs_conversion = any(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line['text']) for line in lines)
        
        if needs_conversion:
            print("🤖 LRCLib found Japanese, converting...")
            converted_lines = await convert_lrc_lines(lines)
            result = {"lyrics": '\n'.join(converted_lines), "type": "lrc_romaji", "source": "lrclib_converted", "ai_used": True}
        else:
            result = {"lyrics": lyrics_text, "type": "lrc_original", "source": "lrclib", "ai_used": False}
            
        save_and_cache(cache_key, result)
        return result
    
    return None

# --- HELPER FUNCTIONS ---

def save_and_cache(key, data):
    song_cache[key] = data
    if redis_client:
        try: redis_client.setex(key, 2592000, json.dumps(data))
        except: pass

def get_from_redis(key):
    if not redis_client: return None
    try:
        data = redis_client.get(key)
        return json.loads(data) if data else None
    except: return None

def parse_lrc_format(lrc_text: str) -> List[Dict]:
    lines = []
    for line in lrc_text.split('\n'):
        if not line.strip(): continue
        match = re.match(r'(\[\d+:\d+\.\d+\])\s*(.*)', line)
        if match:
            lines.append({'timestamp': match.group(1), 'text': match.group(2).strip()})
        else:
            lines.append({'timestamp': '', 'text': line.strip()})
    return lines

async def convert_lrc_lines(lines: List[Dict]) -> List[str]:
    texts = [l['text'] for l in lines if l['text']]
    # DeepSeek V3 has a huge context window, we can send larger batches
    BATCH_SIZE = 50 
    romaji_results = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        converted_batch = await convert_batch(batch)
        romaji_results.extend(converted_batch)
    
    romaji_map = dict(zip(texts, romaji_results))
    result_lines = []
    for line in lines:
        if line['text']:
            romaji = romaji_map.get(line['text'], line['text'])
            result_lines.append(f"{line['timestamp']} {romaji}")
        else:
            result_lines.append("")
    return result_lines

async def convert_japanese_to_romaji(text: str) -> str:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    BATCH_SIZE = 50
    all_converted = []
    for i in range(0, len(lines), BATCH_SIZE):
        batch = lines[i:i+BATCH_SIZE]
        converted = await convert_batch(batch)
        all_converted.extend(converted)
    return '\n'.join(all_converted)

# --- AI CONVERSION ---

def check_quota() -> bool:
    """Non-blocking quota tracker"""
    if not redis_client: return True
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        quota_key = f"quota:deepseek:{today}"
        # We allow it to pass even if 'limit' is reached because limit is 1 Billion
        pipe = redis_client.pipeline()
        pipe.incr(quota_key)
        pipe.expire(quota_key, 86400)
        pipe.execute()
        return True
    except: return True

async def convert_batch(texts: List[str]) -> List[str]:
    if not texts or not client: return texts
    
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

    check_quota() # Track stats but don't block
    
    subset = [texts[i] for i in uncached_idxs]
    
    prompt = f"""Convert these {len(subset)} Japanese lyric lines to Romaji (Hepburn style).
Output EXACTLY {len(subset)} lines. Format: "1. romaji_text"
NO explanations.

INPUT:
{chr(10).join([f"{i+1}. {t}" for i, t in enumerate(subset)])}"""
    
    try:
        completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a Japanese-to-Romaji converter."},
                {"role": "user", "content": prompt}
            ],
            model=DEEPSEEK_MODEL,
            temperature=0.1,
            max_tokens=4096
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
                    if redis_client:
                        try: redis_client.setex(f"line:{thash}", 2592000, json.dumps({'romaji': romaji}))
                        except: pass
                        
    except Exception as e:
        print(f"❌ DeepSeek Error: {e}")

    for i in range(len(results)):
        if results[i] is None: results[i] = texts[i]
        
    return results

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"status": "Online", "provider": "DeepSeek", "model": DEEPSEEK_MODEL}

@app.get("/convert")
async def convert_single_line(text: str = ""):
    if not text: raise HTTPException(400, "No text provided")
    results = await convert_batch([text])
    return {"original": text, "romaji": results[0]}

@app.get("/get_song")
async def get_song_endpoint(song: str, artist: str, duration: int = 0):
    result = await fetch_lyrics_smart(song, artist, duration)
    if result: return result
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
        "limit": "UNLIMITED",
        "model": DEEPSEEK_MODEL
    }
