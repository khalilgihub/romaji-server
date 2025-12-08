from fastapi import FastAPI, HTTPException, BackgroundTasks
import google.generativeai as genai
import requests
import os
import re
import asyncio
from datetime import datetime
from typing import Dict, List

app = FastAPI()

# --- 1. SETUP AI ---
API_KEY = os.environ.get("GEMINI_API_KEY")
model = None

def setup_ai():
    global model
    if not API_KEY:
        print("CRITICAL ERROR: GEMINI_API_KEY is missing!")
        return

    genai.configure(api_key=API_KEY)
    found_model_name = None

    try:
        print("Searching for available AI models...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name:
                    found_model_name = m.name
                    break
                elif 'pro' in m.name and not found_model_name:
                    found_model_name = m.name
        
        if not found_model_name:
            found_model_name = 'models/gemini-pro'

        print(f"--- SUCCESS: Using Model '{found_model_name}' ---")
        model = genai.GenerativeModel(found_model_name)

    except Exception as e:
        print(f"Error finding model: {e}")
        model = genai.GenerativeModel('gemini-pro')

setup_ai()

# --- 2. STORAGE ---
line_cache = {}  # Cache for individual lines
song_cache = {}  # Cache for complete songs: {(song, artist): {lyrics, timestamp}}
processing_queue = []  # Songs currently being processed
popular_songs = []  # List to auto-prepare

# --- 3. POPULAR JAPANESE SONGS LIST ---
POPULAR_JAPANESE_SONGS = [
    ("First Love", "Utada Hikaru"),
    ("Lemon", "Kenshi Yonezu"),
    ("Pretender", "Official HIGE DANdism"),
    ("Kaiju no Hanauta", "Vaundy"),
    ("Shinunoga E-Wa", "Fujii Kaze"),
    ("Dry Flower", "Yuuri"),
    ("Gurenge", "LiSA"),
    ("Homura", "LiSA"),
    ("STAY GOLD", "BTS"),
    ("Bling-Bang-Bang-Born", "Creepy Nuts"),
]

# --- 4. HELPER FUNCTIONS ---
def extract_lyrics_lines(raw_lyrics):
    """Extract lines with their timestamps"""
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
        else:
            if line:
                lines.append({'timestamp': '', 'text': line})
    
    return lines

async def convert_batch_with_retry(texts, max_retries=2):
    """Convert multiple lines with retry logic"""
    if not texts:
        return []
    
    uncached_indices = []
    results = [None] * len(texts)
    
    for i, text in enumerate(texts):
        if text in line_cache:
            results[i] = line_cache[text]
        else:
            uncached_indices.append(i)
    
    if not uncached_indices:
        return results
    
    uncached_texts = [texts[i] for i in uncached_indices]
    batch_text = '\n'.join([f"{i+1}. {text}" for i, text in enumerate(uncached_texts)])
    
    prompt = (
        f"Convert these {len(uncached_texts)} Japanese lines to Romaji (Hepburn). "
        f"Output EXACTLY {len(uncached_texts)} lines numbered 1-{len(uncached_texts)}. "
        f"Format: '1. romaji text'\n\n{batch_text}"
    )
    
    for attempt in range(max_retries + 1):
        try:
            response = await model.generate_content_async(prompt)
            converted_lines = response.text.strip().split('\n')
            
            for line in converted_lines:
                line = line.strip()
                match = re.match(r'^\d+[\.\)]\s*(.+)', line)
                if match:
                    romaji = match.group(1).strip()
                    if romaji:
                        for idx in uncached_indices[:]:
                            if results[idx] is None:
                                original = texts[idx]
                                results[idx] = romaji
                                line_cache[original] = romaji
                                uncached_indices.remove(idx)
                                break
            
            if not uncached_indices:
                break
                
        except Exception as e:
            print(f"Batch error (attempt {attempt + 1}): {e}")
            if attempt == max_retries:
                for idx in uncached_indices:
                    results[idx] = texts[idx]
    
    for i in range(len(results)):
        if results[i] is None:
            results[i] = texts[i]
    
    return results

# --- 5. SONG PREPARATION LOGIC ---
async def prepare_song_internal(song: str, artist: str):
    """Internal function to fetch and convert a song"""
    cache_key = (song.lower(), artist.lower())
    
    # Check if already cached
    if cache_key in song_cache:
        print(f"✓ Song already cached: {song} - {artist}")
        return song_cache[cache_key]
    
    # Check if currently processing
    if cache_key in processing_queue:
        print(f"⏳ Song already in queue: {song} - {artist}")
        return None
    
    processing_queue.append(cache_key)
    print(f"🎵 Starting conversion: {song} - {artist}")
    
    try:
        # Fetch lyrics
        url = "https://lrclib.net/api/get"
        resp = requests.get(url, params={"track_name": song, "artist_name": artist}, timeout=10)
        
        if resp.status_code != 200:
            print(f"❌ Lyrics not found: {song} - {artist}")
            processing_queue.remove(cache_key)
            return None
            
        data = resp.json()
        raw_lyrics = data.get("syncedLyrics") or data.get("plainLyrics")
        
        if not raw_lyrics:
            print(f"❌ Empty lyrics: {song} - {artist}")
            processing_queue.remove(cache_key)
            return None
        
        # Parse and convert
        lyrics_lines = extract_lyrics_lines(raw_lyrics)
        texts_to_convert = [line['text'] for line in lyrics_lines]
        
        # Convert in batches
        BATCH_SIZE = 20
        all_romaji = []
        
        for i in range(0, len(texts_to_convert), BATCH_SIZE):
            batch = texts_to_convert[i:i + BATCH_SIZE]
            romaji_batch = await convert_batch_with_retry(batch)
            all_romaji.extend(romaji_batch)
            
            if i + BATCH_SIZE < len(texts_to_convert):
                await asyncio.sleep(0.3)
        
        # Reconstruct lyrics
        result_lyrics = []
        for i, line_data in enumerate(lyrics_lines):
            if line_data['timestamp']:
                result_lyrics.append(f"{line_data['timestamp']} {all_romaji[i]}")
            else:
                result_lyrics.append(all_romaji[i])
        
        # Cache the result
        result = {
            "lyrics": '\n'.join(result_lyrics),
            "total_lines": len(lyrics_lines),
            "cached_at": datetime.now().isoformat(),
            "song": song,
            "artist": artist
        }
        
        song_cache[cache_key] = result
        print(f"✅ Conversion complete: {song} - {artist} ({len(lyrics_lines)} lines)")
        
        processing_queue.remove(cache_key)
        return result
        
    except Exception as e:
        print(f"❌ Error preparing song {song} - {artist}: {e}")
        if cache_key in processing_queue:
            processing_queue.remove(cache_key)
        return None

# --- 6. BACKGROUND PREPARATION ---
async def prepare_popular_songs():
    """Background task to prepare popular songs"""
    print("🚀 Starting background song preparation...")
    
    for song, artist in POPULAR_JAPANESE_SONGS:
        await prepare_song_internal(song, artist)
        await asyncio.sleep(2)  # Avoid rate limiting
    
    print("✅ Popular songs preparation complete!")

@app.on_event("startup")
async def startup_event():
    """Run when server starts"""
    if model:
        # Start background preparation
        asyncio.create_task(prepare_popular_songs())

# --- 7. API ENDPOINTS ---

@app.get("/")
async def root():
    return {
        "status": "Romaji Converter API Running",
        "model": model.model_name if model else "Not Configured",
        "cached_songs": len(song_cache),
        "cached_lines": len(line_cache)
    }

@app.get("/convert")
async def convert_romaji(text: str = ""):
    """Convert single line to romaji"""
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    if text in line_cache:
        return {"original": text, "romaji": line_cache[text], "cached": True}

    if not model:
        return {"original": text, "romaji": text, "cached": False}

    try:
        results = await convert_batch_with_retry([text])
        return {"original": text, "romaji": results[0], "cached": False}
    except Exception as e:
        print(f"Conversion error: {e}")
        return {"original": text, "romaji": text, "cached": False}

@app.get("/get_song")
async def get_song(song: str, artist: str, background_tasks: BackgroundTasks):
    """Get song lyrics - instant if cached, else prepare in background"""
    cache_key = (song.lower(), artist.lower())
    
    # If already cached, return immediately
    if cache_key in song_cache:
        result = song_cache[cache_key]
        return {
            "status": "ready",
            "cached": True,
            **result
        }
    
    # If not cached, prepare it now
    result = await prepare_song_internal(song, artist)
    
    if result:
        return {
            "status": "ready",
            "cached": False,
            **result
        }
    else:
        raise HTTPException(status_code=404, detail="Song not found or conversion failed")

@app.get("/prepare_song")
async def prepare_song_endpoint(song: str, artist: str):
    """Alias for get_song (backward compatibility)"""
    return await get_song(song, artist, None)

@app.post("/add_to_queue")
async def add_to_queue(songs: List[Dict[str, str]], background_tasks: BackgroundTasks):
    """Add multiple songs to preparation queue"""
    async def process_queue(song_list):
        for item in song_list:
            await prepare_song_internal(item['song'], item['artist'])
            await asyncio.sleep(1)
    
    background_tasks.add_task(process_queue, songs)
    
    return {
        "status": "queued",
        "count": len(songs),
        "message": "Songs added to background processing queue"
    }

@app.get("/cache_status")
async def cache_status():
    """Get cache statistics"""
    return {
        "cached_songs": len(song_cache),
        "cached_lines": len(line_cache),
        "processing_queue": len(processing_queue),
        "songs_list": [f"{s} - {a}" for (s, a) in song_cache.keys()],
        "currently_processing": [f"{s} - {a}" for (s, a) in processing_queue]
    }

@app.get("/search_and_prepare")
async def search_and_prepare(query: str, background_tasks: BackgroundTasks):
    """Search for songs and prepare them"""
    try:
        # Search lrclib for songs
        url = "https://lrclib.net/api/search"
        resp = requests.get(url, params={"q": query}, timeout=10)
        results = resp.json()
        
        if not results:
            raise HTTPException(status_code=404, detail="No songs found")
        
        # Take top 5 results
        songs_to_prepare = []
        for item in results[:5]:
            song = item.get('trackName', '')
            artist = item.get('artistName', '')
            if song and artist:
                songs_to_prepare.append({"song": song, "artist": artist})
        
        # Prepare first song immediately, others in background
        if songs_to_prepare:
            first_result = await prepare_song_internal(
                songs_to_prepare[0]['song'], 
                songs_to_prepare[0]['artist']
            )
            
            # Queue the rest
            if len(songs_to_prepare) > 1:
                async def process_remaining():
                    for item in songs_to_prepare[1:]:
                        await prepare_song_internal(item['song'], item['artist'])
                        await asyncio.sleep(1)
                
                background_tasks.add_task(process_remaining)
            
            return {
                "status": "ready",
                "first_song": first_result,
                "queued_count": len(songs_to_prepare) - 1,
                "all_results": songs_to_prepare
            }
        
        raise HTTPException(status_code=404, detail="No valid songs found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
