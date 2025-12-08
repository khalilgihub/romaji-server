from fastapi import FastAPI, HTTPException, BackgroundTasks
import google.generativeai as genai
import requests
import os
import re
import asyncio
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
import json

app = FastAPI()

# --- 1. SETUP AI ---
API_KEY = os.environ.get("GEMINI_API_KEY")
REDIS_URL = os.environ.get("REDIS_URL")  # Optional but HIGHLY recommended
model = None
redis_client = None

def setup_ai():
    global model
    if not API_KEY:
        print("❌ CRITICAL ERROR: GEMINI_API_KEY is missing!")
        return

    genai.configure(api_key=API_KEY)
    
    try:
        print("🔍 Searching for available AI models...")
        available_models = list(genai.list_models())
        
        # Prioritize: flash-2.0 > flash > pro
        model_priority = ['flash-2.0', 'flash', 'pro']
        found_model_name = None
        
        for priority in model_priority:
            for m in available_models:
                if 'generateContent' in m.supported_generation_methods and priority in m.name.lower():
                    found_model_name = m.name
                    break
            if found_model_name:
                break
        
        if not found_model_name:
            found_model_name = 'models/gemini-pro'

        print(f"✅ SUCCESS: Using Model '{found_model_name}'")
        
        # Configure with better parameters
        generation_config = {
            'temperature': 0.1,  # Lower = more consistent
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 8192,
        }
        
        model = genai.GenerativeModel(
            found_model_name,
            generation_config=generation_config
        )

    except Exception as e:
        print(f"❌ Error finding model: {e}")
        model = genai.GenerativeModel('gemini-pro')

def setup_redis():
    global redis_client
    if not REDIS_URL:
        print("⚠️ REDIS_URL not set - using memory cache only (will reset on restart)")
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

# --- 2. DUAL-LAYER STORAGE ---
# Layer 1: In-memory for speed
line_cache = {}
song_cache = {}

# Layer 2: Redis for persistence
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
    """Save to Redis (default TTL: 30 days)"""
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
    
    # Try memory first
    if cache_key in song_cache:
        return song_cache[cache_key]
    
    # Try Redis
    persistent_data = get_from_persistent(cache_key)
    if persistent_data:
        song_cache[cache_key] = persistent_data  # Populate memory
        return persistent_data
    
    return None

def save_song_cached(song: str, artist: str, data: dict):
    """Save song to both memory AND Redis"""
    cache_key = f"{song.lower()}::{artist.lower()}"
    song_cache[cache_key] = data
    save_to_persistent(cache_key, data)

# Processing queue
processing_queue = set()

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
    ("Bling-Bang-Bang-Born", "Creepy Nuts"),
    ("Shanti", "Ado"),
]

# --- 4. IMPROVED AI CONVERSION ---
async def convert_batch_with_retry(texts: List[str], max_retries: int = 3) -> List[str]:
    """Convert multiple lines with improved retry logic and validation"""
    if not texts or not model:
        return texts
    
    # Check cache first
    uncached_indices = []
    results = [None] * len(texts)
    
    for i, text in enumerate(texts):
        # Create hash for consistent caching
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text in line_cache:
            results[i] = line_cache[text]
        else:
            # Try Redis
            cached_romaji = get_from_persistent(text_hash, prefix="line")
            if cached_romaji:
                romaji_text = cached_romaji.get('romaji', text)
                results[i] = romaji_text
                line_cache[text] = romaji_text
            else:
                uncached_indices.append(i)
    
    if not uncached_indices:
        return results
    
    uncached_texts = [texts[i] for i in uncached_indices]
    
    # Improved prompt for better accuracy
    prompt = f"""Convert these {len(uncached_texts)} Japanese song lyrics to Romaji using Hepburn romanization.

RULES:
- Output EXACTLY {len(uncached_texts)} numbered lines (1 to {len(uncached_texts)})
- Use proper Hepburn romanization (e.g., 'shi' not 'si', 'tsu' not 'tu')
- Keep particles natural (wa, e, o)
- Preserve punctuation and spacing
- No explanations, only the conversions

INPUT:
{chr(10).join([f"{i+1}. {text}" for i, text in enumerate(uncached_texts)])}

OUTPUT (numbered 1-{len(uncached_texts)}):"""
    
    for attempt in range(max_retries + 1):
        try:
            response = await asyncio.wait_for(
                model.generate_content_async(prompt),
                timeout=30.0
            )
            
            converted_lines = response.text.strip().split('\n')
            parsed_count = 0
            
            for line in converted_lines:
                line = line.strip()
                if not line:
                    continue
                
                # Match numbered lines more flexibly
                match = re.match(r'^(\d+)[\.\):\-\s]+(.+)', line)
                if match:
                    line_num = int(match.group(1))
                    romaji = match.group(2).strip()
                    
                    # Validate line number
                    if 1 <= line_num <= len(uncached_texts):
                        idx = uncached_indices[line_num - 1]
                        if results[idx] is None:
                            results[idx] = romaji
                            original = texts[idx]
                            line_cache[original] = romaji
                            
                            # Save to Redis
                            text_hash = hashlib.md5(original.encode()).hexdigest()
                            save_to_persistent(text_hash, {'romaji': romaji}, prefix="line")
                            
                            parsed_count += 1
            
            # Check if we got all conversions
            if all(results[i] is not None for i in uncached_indices):
                print(f"✅ Batch conversion successful: {len(uncached_texts)} lines")
                break
            
            if attempt < max_retries:
                print(f"⚠️ Retry {attempt + 1}: Only got {parsed_count}/{len(uncached_texts)} lines")
                await asyncio.sleep(1)
                
        except asyncio.TimeoutError:
            print(f"⏱️ Timeout on attempt {attempt + 1}")
            if attempt < max_retries:
                await asyncio.sleep(2)
        except Exception as e:
            print(f"❌ Batch error (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                await asyncio.sleep(2)
    
    # Fill any remaining None values with original text
    for i in range(len(results)):
        if results[i] is None:
            results[i] = texts[i]
            print(f"⚠️ Failed to convert line {i+1}: {texts[i][:50]}...")
    
    return results

# --- 5. HELPER FUNCTIONS ---
def extract_lyrics_lines(raw_lyrics: str) -> List[dict]:
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
        elif line:
            lines.append({'timestamp': '', 'text': line})
    
    return lines

async def fetch_lyrics_async(song: str, artist: str) -> Optional[str]:
    """Fetch lyrics asynchronously"""
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

# --- 6. SONG PREPARATION LOGIC ---
async def prepare_song_internal(song: str, artist: str) -> Optional[dict]:
    """Internal function to fetch and convert a song"""
    cache_key = f"{song.lower()}::{artist.lower()}"
    
    # Check cache
    cached = get_song_cached(song, artist)
    if cached:
        print(f"✓ Cache hit: {song} - {artist}")
        return cached
    
    # Check if currently processing
    if cache_key in processing_queue:
        print(f"⏳ Already processing: {song} - {artist}")
        # Wait for it to finish (with timeout)
        for _ in range(30):  # Wait up to 30 seconds
            await asyncio.sleep(1)
            cached = get_song_cached(song, artist)
            if cached:
                return cached
        return None
    
    processing_queue.add(cache_key)
    print(f"🎵 Starting conversion: {song} - {artist}")
    
    try:
        # Fetch lyrics
        raw_lyrics = await fetch_lyrics_async(song, artist)
        
        if not raw_lyrics:
            print(f"❌ Lyrics not found: {song} - {artist}")
            processing_queue.discard(cache_key)
            return None
        
        # Parse lyrics
        lyrics_lines = extract_lyrics_lines(raw_lyrics)
        if not lyrics_lines:
            print(f"❌ Empty lyrics: {song} - {artist}")
            processing_queue.discard(cache_key)
            return None
        
        texts_to_convert = [line['text'] for line in lyrics_lines]
        
        # Convert in optimized batches
        BATCH_SIZE = 25  # Increased from 20
        all_romaji = []
        
        for i in range(0, len(texts_to_convert), BATCH_SIZE):
            batch = texts_to_convert[i:i + BATCH_SIZE]
            romaji_batch = await convert_batch_with_retry(batch)
            all_romaji.extend(romaji_batch)
            
            # Rate limiting between batches
            if i + BATCH_SIZE < len(texts_to_convert):
                await asyncio.sleep(0.5)
        
        # Reconstruct lyrics with timestamps
        result_lyrics = []
        for i, line_data in enumerate(lyrics_lines):
            if line_data['timestamp']:
                result_lyrics.append(f"{line_data['timestamp']} {all_romaji[i]}")
            else:
                result_lyrics.append(all_romaji[i])
        
        # Create result
        result = {
            "lyrics": '\n'.join(result_lyrics),
            "total_lines": len(lyrics_lines),
            "cached_at": datetime.now().isoformat(),
            "song": song,
            "artist": artist
        }
        
        # Save to cache
        save_song_cached(song, artist, result)
        print(f"✅ Conversion complete: {song} - {artist} ({len(lyrics_lines)} lines)")
        
        processing_queue.discard(cache_key)
        return result
        
    except Exception as e:
        print(f"❌ Error preparing song {song} - {artist}: {e}")
        processing_queue.discard(cache_key)
        return None

# --- 7. BACKGROUND PREPARATION ---
async def prepare_popular_songs():
    """Background task to prepare popular songs"""
    print("🚀 Starting background song preparation...")
    
    for song, artist in POPULAR_JAPANESE_SONGS:
        try:
            await prepare_song_internal(song, artist)
            await asyncio.sleep(3)  # Rate limiting
        except Exception as e:
            print(f"Error preparing {song}: {e}")
    
    print("✅ Popular songs preparation complete!")

@app.on_event("startup")
async def startup_event():
    """Run when server starts"""
    if model:
        asyncio.create_task(prepare_popular_songs())

# --- 8. API ENDPOINTS ---

@app.get("/")
async def root():
    """Server status"""
    return {
        "status": "Romaji Converter API Running",
        "model": model.model_name if model else "Not Configured",
        "cached_songs_memory": len(song_cache),
        "cached_lines_memory": len(line_cache),
        "redis_connected": redis_client is not None,
        "currently_processing": len(processing_queue)
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
async def get_song(song: str, artist: str):
    """Get song lyrics - instant if cached, else convert now"""
    result = await prepare_song_internal(song, artist)
    
    if result:
        return {
            "status": "ready",
            **result
        }
    else:
        raise HTTPException(status_code=404, detail="Song not found or conversion failed")

@app.post("/add_to_queue")
async def add_to_queue(songs: List[Dict[str, str]], background_tasks: BackgroundTasks):
    """Add multiple songs to preparation queue"""
    async def process_queue(song_list):
        for item in song_list:
            try:
                await prepare_song_internal(item.get('song', ''), item.get('artist', ''))
                await asyncio.sleep(2)
            except Exception as e:
                print(f"Queue processing error: {e}")
    
    background_tasks.add_task(process_queue, songs)
    
    return {
        "status": "queued",
        "count": len(songs),
        "message": "Songs added to background processing queue"
    }

@app.get("/cache_status")
async def cache_status():
    """Get cache statistics"""
    redis_song_count = 0
    if redis_client:
        try:
            redis_song_count = len([k for k in redis_client.keys("song:*")])
        except:
            pass
    
    return {
        "memory": {
            "cached_songs": len(song_cache),
            "cached_lines": len(line_cache),
            "processing_queue": len(processing_queue)
        },
        "redis": {
            "connected": redis_client is not None,
            "cached_songs": redis_song_count
        },
        "songs_list": [f"{s} - {a}" for s, a in 
                      [(v['song'], v['artist']) for v in song_cache.values()]],
        "currently_processing": list(processing_queue)
    }

@app.get("/search_and_prepare")
async def search_and_prepare(query: str, background_tasks: BackgroundTasks):
    """Search for songs and prepare them"""
    try:
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
        
        if not songs_to_prepare:
            raise HTTPException(status_code=404, detail="No valid songs found")
        
        # Prepare first song immediately
        first_result = await prepare_song_internal(
            songs_to_prepare[0]['song'], 
            songs_to_prepare[0]['artist']
        )
        
        # Queue the rest in background
        if len(songs_to_prepare) > 1:
            async def process_remaining():
                for item in songs_to_prepare[1:]:
                    await prepare_song_internal(item['song'], item['artist'])
                    await asyncio.sleep(2)
            
            background_tasks.add_task(process_remaining)
        
        return {
            "status": "ready",
            "first_song": first_result,
            "queued_count": len(songs_to_prepare) - 1,
            "all_results": songs_to_prepare
        }
        
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Search API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear_cache")
async def clear_cache():
    """Clear all caches (use with caution)"""
    line_cache.clear()
    song_cache.clear()
    
    redis_cleared = 0
    if redis_client:
        try:
            redis_cleared = redis_client.delete(*redis_client.keys("song:*"))
            redis_cleared += redis_client.delete(*redis_client.keys("line:*"))
        except Exception as e:
            print(f"Redis clear error: {e}")
    
    return {
        "status": "cleared",
        "memory_cleared": True,
        "redis_keys_deleted": redis_cleared
    }
