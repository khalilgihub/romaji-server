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

DEEPSEEK_MODEL = "deepseek-chat" 

client = None
redis_client = None
song_cache = {}

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
    """Get the perfect timestamps from LRCLib"""
    try:
        url = "https://lrclib.net/api/get"
        params = {"track_name": song, "artist_name": artist}
        if duration > 0: params["duration"] = duration
        
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        # We specifically want SYNCED lyrics for the timestamps
        lrc_text = data.get("syncedLyrics")
        if not lrc_text: return None
        
        lines = []
        for line in lrc_text.split('\n'):
            if not line.strip(): continue
            # Regex to grab [00:12.34] and the Japanese text
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
    """Search Genius specifically for 'Song Name Romaji'"""
    if not GENIUS_API_TOKEN: return None
    try:
        headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
        
        # 1. Search for "Song Title (Romaji)"
        search_query = f"{song} Romaji {artist}"
        resp = requests.get("https://api.genius.com/search", headers=headers, params={"q": search_query}, timeout=10)
        data = resp.json()
        
        hit_url = None
        if data['response']['hits']:
            hit_url = data['response']['hits'][0]['result']['url']
            print(f"🔎 Genius Hit: {data['response']['hits'][0]['result']['full_title']}")
        else:
            # Fallback: Search normal title, sometimes Romaji is inside
            resp = requests.get("https://api.genius.com/search", headers=headers, params={"q": f"{song} {artist}"}, timeout=10)
            data = resp.json()
            if data['response']['hits']:
                hit_url = data['response']['hits'][0]['result']['url']

        if not hit_url: return None

        # 2. Scrape the text
        page = requests.get(hit_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        soup = BeautifulSoup(page.text, 'html.parser')
        
        lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
        full_text = []
        for div in lyrics_divs:
            for el in div.descendants:
                if el.name == 'br': full_text.append('\n')
                elif isinstance(el, str):
                    t = el.strip()
                    if t and t[0] != '[': # Skip [Chorus], [Verse] tags
                         full_text.append(t)
        
        romaji_block = '\n'.join(full_text)
        
        # 3. Validation: Does it look like Romaji? (ASCII characters)
        if re.match(r'^[a-zA-Z\s\-\'\.!?,()]+$', romaji_block[:100]):
            return romaji_block
        
        # If it looks like Japanese, this failed.
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', romaji_block[:50]):
            print("⚠️ Genius result was Japanese, not Romaji.")
            return None
            
        return romaji_block

    except Exception as e:
        print(f"Genius Error: {e}")
        return None

# --- 3. THE MERGER (AI Logic) ---
async def align_lyrics(lrc_lines: List[Dict], romaji_block: str) -> List[str]:
    """Asks AI to paste the Romaji onto the Timestamps"""
    if not client: return []
    
    # We send the structure to the AI
    prompt = f"""You are a Lyrics Aligner.
    
    TASK:
    I have a list of TIMESTAMPS with original JAPANESE text.
    I have a block of ROMAJI text that matches the song.
    
    Your job is to replace the Japanese text with the corresponding Romaji line, keeping the timestamp exactly the same.
    
    RULES:
    1. If there are more timestamps than Romaji lines, translate the remaining Japanese yourself.
    2. If lines don't match perfectly, align them as best as possible based on meaning/sound.
    3. Output STRICT JSON format.
    
    TIMESTAMPS & JAPANESE:
    {json.dumps(lrc_lines[:50], ensure_ascii=False)} 
    (List truncated to 50 for context, but process logically)
    
    ROMAJI SOURCE:
    {romaji_block[:2000]}
    
    OUTPUT SCHEMA:
    {{
        "lines": ["00:12.34 Romaji Text", "00:15.67 Romaji Text"]
    }}
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

# --- 4. FALLBACK TRANSLATOR (If Genius fails) ---
async def just_translate(lrc_lines: List[Dict]) -> List[str]:
    print("🤖 Genius Romaji not found. Translating raw Japanese...")
    if not client: return []
    
    # Batch the Japanese text
    texts = [l['japanese'] for l in lrc_lines]
    
    prompt = f"""Convert these {len(texts)} Japanese lines to Romaji.
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
        romaji_list = data.get("romaji", [])
        
        # Stitch back timestamps
        final = []
        for i, r in enumerate(romaji_list):
            if i < len(lrc_lines):
                final.append(f"{lrc_lines[i]['timestamp']} {r}")
        return final
    except: return []

# --- MAIN LOGIC FLOW ---
async def process_song(song: str, artist: str, duration: int = 0):
    cache_key = f"merged:{song.lower()}:{artist.lower()}"
    
    # 1. Check Cache
    if cache_key in song_cache: return song_cache[cache_key]
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached: 
            data = json.loads(cached)
            song_cache[cache_key] = data
            return data

    print(f"🚀 Processing: {song}...")

    # 2. Get Timestamps (REQUIRED)
    lrc_data = fetch_lrc_timestamps(song, artist, duration)
    if not lrc_data:
        raise HTTPException(404, "No Timed Lyrics (LRC) found on LRCLib.")

    # 3. Try to get Romaji from Genius
    romaji_text = fetch_genius_romaji(song, artist)
    
    final_lyrics = []
    
    if romaji_text:
        print("✨ Found Romaji on Genius! merging...")
        # 4a. MERGE STRATEGY
        final_lyrics = await align_lyrics(lrc_data, romaji_text)
        source_tag = "Merged (Genius Text + LRCLib Time)"
    else:
        # 4b. TRANSLATE STRATEGY
        final_lyrics = await just_translate(lrc_data)
        source_tag = "Translated (AI)"
        
    if not final_lyrics:
        raise HTTPException(500, "AI failed to process lyrics.")

    result = {
        "lyrics": '\n'.join(final_lyrics),
        "song": song,
        "source": source_tag
    }
    
    # 5. Save
    song_cache[cache_key] = result
    if redis_client:
        redis_client.setex(cache_key, 2592000, json.dumps(result))
        
    return result

# --- ENDPOINTS ---
@app.get("/get_song")
async def get_song_endpoint(song: str, artist: str, duration: int = 0):
    return await process_song(song, artist, duration)

@app.get("/")
async def root():
    return {"status": "Online", "mode": "Merger (Genius+LRCLib)"}

@app.delete("/clear_cache")
async def clear():
    song_cache.clear()
    if redis_client: redis_client.flushdb()
    return {"status": "cleared"}
