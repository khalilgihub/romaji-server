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
line_cache = {}

def setup_systems():
    global client, redis_client
    if DEEPSEEK_API_KEY:
        try:
            client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            print(f"✅ DeepSeek AI Online: {DEEPSEEK_MODEL}")
        except: pass
    if REDIS_URL:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            redis_client.ping()
            print("✅ Redis Online")
        except: pass
    if GENIUS_API_TOKEN:
        print("✅ Genius API Token Loaded")

setup_systems()

# --- 1. FETCH TIMESTAMPS (LRCLib) ---
async def fetch_lrc_timestamps(song: str, artist: str) -> Optional[List[Dict]]:
    try:
        url = "https://lrclib.net/api/get"
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: requests.get(url, params={"track_name": song, "artist_name": artist}, timeout=10))
        data = resp.json()
        lrc_text = data.get("syncedLyrics")
        if not lrc_text: return None
        
        lines = []
        for line in lrc_text.split('\n'):
            if not line.strip(): continue
            match = re.match(r'(\[\d+:\d+\.\d+\])\s*(.*)', line)
            if match:
                # We keep the Japanese text as a reference/fallback
                lines.append({'timestamp': match.group(1), 'reference': match.group(2).strip()})
        return lines
    except: return None

# --- 2. FETCH ROMAJI (Genius) ---
async def fetch_genius_romaji(song: str, artist: str) -> Optional[str]:
    if not GENIUS_API_TOKEN: return None
    try:
        headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
        loop = asyncio.get_event_loop()
        
        search_query = f"{song} Romaji {artist}"
        resp = await loop.run_in_executor(None, lambda: requests.get("https://api.genius.com/search", headers=headers, params={"q": search_query}, timeout=10))
        data = resp.json()
        
        hit_url = None
        if data['response']['hits']:
            hit_url = data['response']['hits'][0]['result']['url']
        
        if not hit_url: return None

        page = await loop.run_in_executor(None, lambda: requests.get(hit_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10))
        soup = BeautifulSoup(page.text, 'html.parser')
        
        lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
        full_text = []
        for div in lyrics_divs:
            for el in div.descendants:
                if el.name == 'br': full_text.append('\n')
                elif isinstance(el, str):
                    t = el.strip()
                    if t and t[0] != '[': full_text.append(t)
        
        text = '\n'.join(full_text)
        
        # Validation: If it has too many Japanese characters, it's not Romaji
        jp_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))
        if jp_chars > len(text) * 0.2: 
            return None
            
        return text
    except: return None

# --- 3. AI ALIGNER (The Smart Merger) ---
async def align_lyrics_strict(lrc_lines: List[Dict], romaji_block: str) -> List[str]:
    if not client: return []
    
    prompt = f"""You are a Lyrics Synchronizer.
    
    I have {len(lrc_lines)} TIMESTAMPS (Japanese source).
    I have a block of ROMAJI TEXT (from Genius).
    
    TASK: Map the Romaji to the Timestamps.
    
    CRITICAL RULES:
    1. Output EXACTLY {len(lrc_lines)} lines. One for each timestamp.
    2. If Genius has extra lines (like [Chorus] or duplicates), SKIP THEM.
    3. If Genius is MISSING lines, TRANSLATE the Japanese reference I gave you.
    4. Priority is SYNC over TEXT. Do not break the timestamp order.
    
    TIMESTAMPS & REFERENCE JP:
    {json.dumps(lrc_lines[:60], ensure_ascii=False)}
    
    ROMAJI SOURCE:
    {romaji_block[:3000]}
    
    OUTPUT SCHEMA JSON:
    {{ "lines": ["[00:12.34] Romaji Text"] }}
    """
    
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        data = json.loads(completion.choices[0].message.content)
        result_lines = data.get("lines", [])
        
        # Safety Check: Did AI return the right amount?
        if len(result_lines) != len(lrc_lines):
            print(f"⚠️ Mismatch detected (Expected {len(lrc_lines)}, got {len(result_lines)}). Aborting merge.")
            return [] # Fail to trigger fallback
            
        return result_lines
    except Exception as e:
        print(f"Alignment Error: {e}")
        return []

# --- 4. FALLBACK TRANSLATOR (Guaranteed Sync) ---
async def just_translate(lrc_lines: List[Dict]) -> List[str]:
    if not client: return []
    texts = [l['reference'] for l in lrc_lines]
    
    prompt = f"""Convert these {len(texts)} Japanese lines to Hepburn Romaji.
    Output JSON: {{ "romaji": ["line1", "line2"] }}
    Strictly 1 line input = 1 line output.
    INPUT: {json.dumps(texts, ensure_ascii=False)}"""
    
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(completion.choices[0].message.content)
        romaji = data.get("romaji", [])
        
        final = []
        for i, r in enumerate(romaji):
            if i < len(lrc_lines):
                final.append(f"{lrc_lines[i]['timestamp']} {r}")
        return final
    except: return []

# --- 5. MAIN LOGIC ---
async def process_song(song: str, artist: str):
    cache_key = f"hybrid:{song.lower()}:{artist.lower()}"
    
    if cache_key in song_cache: return song_cache[cache_key]
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached: return json.loads(cached)

    print(f"🚀 Processing: {song}...")
    
    # 1. Get Timestamps (Required)
    lrc_data = await fetch_lrc_timestamps(song, artist)
    if not lrc_data:
        raise HTTPException(404, "No Timestamps found (LRCLib)")

    # 2. Try to get Genius Romaji
    romaji_text = await fetch_genius_romaji(song, artist)
    
    final_lyrics = []
    source = ""

    if romaji_text:
        print("✨ Found Genius Romaji! Attempting strict align...")
        final_lyrics = await align_lyrics_strict(lrc_data, romaji_text)
        
        if final_lyrics:
            source = "Merged (Genius Romaji)"
        else:
            print("⚠️ Alignment unsafe. Falling back to Translation.")
            final_lyrics = await just_translate(lrc_data)
            source = "Translated (AI Fallback)"
    else:
        print("🤖 No Genius Romaji found. Translating...")
        final_lyrics = await just_translate(lrc_data)
        source = "Translated (AI)"
        
    result = {"lyrics": '\n'.join(final_lyrics), "song": song, "source": source}
    
    song_cache[cache_key] = result
    if redis_client: redis_client.setex(cache_key, 2592000, json.dumps(result))
        
    return result

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {"status": "Online", "mode": "Safe Hybrid"}

# ✅ FIXED: Restored for Floating App compatibility
@app.get("/convert")
async def convert_single_line(text: str = ""):
    if not text: raise HTTPException(400, "No text")
    if text in line_cache: return {"original": text, "romaji": line_cache[text]}
    
    if not client: return {"original": text, "romaji": text}
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": f"Convert to Romaji: {text}"}],
            model=DEEPSEEK_MODEL
        )
        romaji = completion.choices[0].message.content.strip()
        line_cache[text] = romaji
        return {"original": text, "romaji": romaji}
    except: return {"original": text, "romaji": text}

@app.get("/get_song")
async def get_song_endpoint(song: str, artist: str):
    return await process_song(song, artist)

@app.delete("/clear_cache")
async def clear():
    song_cache.clear()
    line_cache.clear()
    if redis_client: redis_client.flushdb()
    return {"status": "cleared"}
