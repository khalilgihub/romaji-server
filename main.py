from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
import requests
import os
import re
import hashlib
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

DEEPSEEK_MODEL = "deepseek-chat" 

client = None
redis_client = None
song_cache = {}
line_cache = {} # Cache for single lines

def setup_systems():
    global client, redis_client
    if DEEPSEEK_API_KEY:
        try:
            client = AsyncOpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com"
            )
            print(f"✅ DeepSeek AI Online: {DEEPSEEK_MODEL}")
        except Exception as e:
            print(f"❌ AI Error: {e}")
    
    if REDIS_URL:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            redis_client.ping()
            print("✅ Redis Online")
        except: pass
            
    if GENIUS_API_TOKEN:
        print("✅ Genius API Token Loaded")

setup_systems()

# --- 1. GET TIMESTAMPS (LRCLib) ---
def fetch_lrc_timestamps(song: str, artist: str, duration: int = 0) -> Optional[List[Dict]]:
    try:
        url = "https://lrclib.net/api/get"
        params = {"track_name": song, "artist_name": artist}
        if duration > 0: params["duration"] = duration
        
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        lrc_text = data.get("syncedLyrics")
        if not lrc_text: return None
        
        lines = []
        for line in lrc_text.split('\n'):
            if not line.strip(): continue
            match = re.match(r'(\[\d+:\d+\.\d+\])\s*(.*)', line)
            if match:
                lines.append({
                    'timestamp': match.group(1), 
                    'japanese': match.group(2).strip()
                })
        return lines
    except: return None

# --- 2. GET ROMAJI TEXT (Genius) ---
def fetch_genius_romaji(song: str, artist: str) -> Optional[str]:
    if not GENIUS_API_TOKEN: return None
    try:
        headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
        
        # Search for "Song Title (Romaji)"
        search_query = f"{song} Romaji {artist}"
        resp = requests.get("https://api.genius.com/search", headers=headers, params={"q": search_query}, timeout=10)
        data = resp.json()
        
        hit_url = None
        if data['response']['hits']:
            hit_url = data['response']['hits'][0]['result']['url']
        else:
            # Fallback search
            resp = requests.get("https://api.genius.com/search", headers=headers, params={"q": f"{song} {artist}"}, timeout=10)
            data = resp.json()
            if data['response']['hits']:
                hit_url = data['response']['hits'][0]['result']['url']

        if not hit_url: return None

        page = requests.get(hit_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        soup = BeautifulSoup(page.text, 'html.parser')
        
        lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
        full_text = []
        for div in lyrics_divs:
            for el in div.descendants:
                if el.name == 'br': full_text.append('\n')
                elif isinstance(el, str):
                    t = el.strip()
                    if t and t[0] != '[': full_text.append(t)
        
        romaji_block = '\n'.join(full_text)
        
        # Validation: Does it look like Romaji?
        if re.match(r'^[a-zA-Z\s\-\'\.!?,()]+$', romaji_block[:100]):
            return romaji_block
        return None

    except Exception as e:
        print(f"Genius Error: {e}")
        return None

# --- 3. THE MERGER (AI Logic) ---
async def align_lyrics(lrc_lines: List[Dict], romaji_block: str) -> List[str]:
    if not client: return []
    
    prompt = f"""You are a Lyrics Aligner.
    TASK: Replace Japanese text with corresponding Romaji line. Keep timestamps.
    
    TIMESTAMPS:
    {json.dumps(lrc_lines[:60], ensure_ascii=False)}
    
    ROMAJI SOURCE:
    {romaji_block[:3000]}
    
    OUTPUT SCHEMA JSON:
    {{ "lines": ["00:12.34 Romaji Text"] }}
    """
    
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        data = json.loads(completion.choices[0].message.content)
        return data.get("lines", [])
    except Exception as e:
        print(f"Alignment Error: {e}")
        return []

# --- 4. FALLBACK TRANSLATOR ---
async def just_translate(texts: List[str]) -> List[str]:
    if not client: return texts
    
    prompt = f"""Convert these Japanese lines to Romaji.
    Output JSON: {{ "romaji": ["line1", "line2"] }}
    INPUT: {json.dumps(texts, ensure_ascii=False)}"""
    
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(completion.choices[0].message.content)
        return data.get("romaji", texts)
    except: return texts

# --- 5. PROCESS SONG ---
async def process_song(song: str, artist: str, duration: int = 0):
    cache_key = f"merged:{song.lower()}:{artist.lower()}"
    
    if cache_key in song_cache: return song_cache[cache_key]
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached: 
            data = json.loads(cached)
            song_cache[cache_key] = data
            return data

    print(f"🚀 Processing: {song}...")
    lrc_data = fetch_lrc_timestamps(song, artist, duration)
    if not lrc_data:
        raise HTTPException(404, "No Timestamps found on LRCLib.")

    romaji_text = fetch_genius_romaji(song, artist)
    final_lyrics = []
    
    if romaji_text:
        print("✨ Merging with Genius...")
        final_lyrics = await align_lyrics(lrc_data, romaji_text)
        source = "Merged (Genius+LRC)"
    else:
        print("🤖 Translating raw Japanese...")
        jp_texts = [x['japanese'] for x in lrc_data]
        romaji_list = await just_translate(jp_texts)
        for i, r in enumerate(romaji_list):
            if i < len(lrc_data):
                final_lyrics.append(f"{lrc_data[i]['timestamp']} {r}")
        source = "Translated (AI)"
        
    result = {
        "lyrics": '\n'.join(final_lyrics),
        "song": song,
        "source": source
    }
    
    song_cache[cache_key] = result
    if redis_client:
        redis_client.setex(cache_key, 2592000, json.dumps(result))
    return result

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"status": "Online", "mode": "Merger + Converter"}

# ✅ RESTORED: This is the fix for your 404 errors!
@app.get("/convert")
async def convert_single_line(text: str = ""):
    if not text: raise HTTPException(400, "No text")
    
    # Check cache
    if text in line_cache: return {"original": text, "romaji": line_cache[text]}
    
    # Convert using AI
    results = await just_translate([text])
    romaji = results[0]
    
    # Save cache
    line_cache[text] = romaji
    return {"original": text, "romaji": romaji}

@app.get("/get_song")
async def get_song_endpoint(song: str, artist: str, duration: int = 0):
    return await process_song(song, artist, duration)

@app.delete("/clear_cache")
async def clear():
    song_cache.clear()
    line_cache.clear()
    if redis_client: redis_client.flushdb()
    return {"status": "cleared"}
