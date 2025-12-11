from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
import requests
import os
import re
import hashlib
import unicodedata
from typing import List, Optional, Dict
import json
import redis
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time
from fastapi.responses import StreamingResponse
from difflib import SequenceMatcher

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

# --- TEXT NORMALIZATION UTILS ---
def normalize_text(text: str) -> str:
    """Normalize text for better matching"""
    text = unicodedata.normalize('NFKC', text.lower())
    text = re.sub(r'[「」【】『』()\[\]{}]', '', text)  # Remove brackets
    text = re.sub(r'[!?.,;:]', '', text)  # Remove punctuation
    return text.strip()

# --- 1. FETCH TIMESTAMPS (LRCLib) ---
def parse_lrc_lines(lrc_text: str) -> List[Dict]:
    """Parse LRC text into structured lines"""
    lines = []
    for line in lrc_text.split('\n'):
        if not line.strip(): 
            continue
        match = re.match(r'(\[\d+:\d+\.\d+\])\s*(.*)', line)
        if match:
            lines.append({
                'timestamp': match.group(1), 
                'reference': match.group(2).strip()
            })
    return lines

async def fetch_lrc_timestamps(song: str, artist: str) -> Optional[List[Dict]]:
    try:
        url = "https://lrclib.net/api/get"
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, 
            lambda: requests.get(
                url, 
                params={"track_name": song, "artist_name": artist}, 
                timeout=5
            )
        )
        data = resp.json()
        lrc_text = data.get("syncedLyrics")
        if not lrc_text: 
            return None
        
        return parse_lrc_lines(lrc_text)
    except: 
        return None

# --- 2. IMPROVED GENIUS FETCHER ---
async def fetch_genius_lyrics_parallel(song: str, artist: str) -> Optional[str]:
    """Get structured romaji lyrics from Genius"""
    if not GENIUS_API_TOKEN: 
        return None
    
    try:
        headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
        loop = asyncio.get_event_loop()
        
        # Search for song
        search_query = f"{song} {artist} romaji"
        resp = await loop.run_in_executor(
            None, 
            lambda: requests.get(
                "https://api.genius.com/search", 
                headers=headers, 
                params={"q": search_query}, 
                timeout=8
            )
        )
        data = resp.json()
        
        if not data['response']['hits']:
            return None
        
        song_url = data['response']['hits'][0]['result']['url']
        
        # Fetch and parse page
        page = await loop.run_in_executor(
            None,
            lambda: requests.get(
                song_url, 
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                }, 
                timeout=8
            )
        )
        soup = BeautifulSoup(page.text, 'html.parser')
        
        # Extract structured lyrics
        lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
        full_text = []
        
        for div in lyrics_divs:
            for el in div.descendants:
                if el.name == 'br':
                    full_text.append('\n')
                elif isinstance(el, str):
                    t = el.strip()
                    if t and not t.startswith('[') and not t.endswith(']'):
                        full_text.append(t)
        
        text = ''.join(full_text)
        
        # Better validation: Check for Romaji patterns
        romaji_pattern = r'\b[a-zA-ZāēīōūĀĒĪŌŪ]+[a-zA-ZāēīōūĀĒĪŌŪ\s]*\b'
        romaji_count = len(re.findall(romaji_pattern, text))
        total_words = len(text.split())
        
        if total_words > 0 and romaji_count / total_words < 0.5:
            return None  # Not enough Romaji
        
        return text
    except Exception as e:
        print(f"Genius fetch error: {e}")
        return None

# --- 3. SMART MATCHING ALGORITHM ---
def find_best_match(japanese_line: str, romaji_candidates: List[str]) -> Optional[str]:
    """Find the best romaji match for a Japanese line"""
    normalized_jp = normalize_text(japanese_line)
    best_score = 0
    best_match = None
    
    for romaji in romaji_candidates:
        normalized_romaji = normalize_text(romaji)
        
        # Check for direct word overlap
        jp_words = set(normalized_jp.split())
        romaji_words = set(normalized_romaji.split())
        overlap = len(jp_words & romaji_words) / max(len(jp_words), 1)
        
        if overlap > 0.3:  # If they share words
            score = SequenceMatcher(None, normalized_jp, normalized_romaji).ratio()
            if score > best_score:
                best_score = score
                best_match = romaji
    
    return best_match if best_score > 0.4 else None

async def smart_align_lyrics(lrc_lines: List[Dict], romaji_text: str) -> List[str]:
    """Intelligently align Japanese timestamps with Romaji lines"""
    
    # Clean and split romaji text
    romaji_lines = []
    for line in romaji_text.split('\n'):
        line = line.strip()
        if line and not re.match(r'^[0-9]+\.[0-9]+$', line):  # Skip timestamps
            romaji_lines.append(line)
    
    # Matching algorithm
    romaji_index = 0
    aligned_result = []
    
    for i, lrc_line in enumerate(lrc_lines):
        japanese_text = lrc_line['reference']
        
        # Look ahead in romaji (sliding window)
        window_size = min(5, len(romaji_lines) - romaji_index)
        candidates = []
        
        for j in range(romaji_index, min(romaji_index + window_size, len(romaji_lines))):
            candidates.append(romaji_lines[j])
        
        if candidates:
            best_match = find_best_match(japanese_text, candidates)
            
            if best_match:
                # Move romaji index to after the matched line
                for j in range(romaji_index, len(romaji_lines)):
                    if romaji_lines[j] == best_match:
                        romaji_index = j + 1
                        break
                aligned_result.append(f"{lrc_line['timestamp']} {best_match}")
                continue
        
        # No match found in window, try broader search
        if romaji_index < len(romaji_lines):
            aligned_result.append(f"{lrc_line['timestamp']} {romaji_lines[romaji_index]}")
            romaji_index += 1
        else:
            # Fallback to translation
            aligned_result.append(f"{lrc_line['timestamp']} {japanese_text}")
    
    return aligned_result

# --- 4. BATCH TRANSLATION (FASTER) ---
async def translate_line(text: str) -> str:
    """Translate a single line quickly"""
    prompt = f"Translate this Japanese to Romaji (Hepburn). Only output the romaji: {text}"
    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=DEEPSEEK_MODEL,
        temperature=0,
        max_tokens=50
    )
    return completion.choices[0].message.content.strip()

async def batch_translate_lines(japanese_lines: List[str]) -> List[str]:
    """Translate multiple lines at once for efficiency"""
    if not client or not japanese_lines:
        return japanese_lines
    
    # Group lines for efficient processing
    batch_size = 20
    results = []
    
    for i in range(0, len(japanese_lines), batch_size):
        batch = japanese_lines[i:i+batch_size]
        prompt = f"""Translate these Japanese lines to Romaji (Hepburn).
        Output as a JSON array with exactly {len(batch)} strings.
        
        Lines: {json.dumps(batch, ensure_ascii=False)}
        
        Output format: {{"translations": ["romaji1", "romaji2", ...]}}"""
        
        try:
            completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=DEEPSEEK_MODEL,
                temperature=0,
                response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content)
            results.extend(data.get("translations", batch))
        except:
            results.extend(batch)  # Fallback
    
    return results

# --- 5. MAIN PROCESSING (OPTIMIZED) ---
async def process_song(song: str, artist: str):
    cache_key = f"hybrid:{song.lower()}:{artist.lower()}"
    
    if cache_key in song_cache: 
        return song_cache[cache_key]
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached: 
            song_cache[cache_key] = json.loads(cached)
            return song_cache[cache_key]

    print(f"🚀 Processing: {song}...")
    start_time = time.time()
    
    # Parallel fetch
    loop = asyncio.get_event_loop()
    
    # Fetch LRC and Genius in parallel
    lrc_future = loop.run_in_executor(
        executor, 
        lambda: requests.get(
            "https://lrclib.net/api/get",
            params={"track_name": song, "artist_name": artist},
            timeout=5
        )
    )
    
    genius_future = None
    if GENIUS_API_TOKEN:
        genius_future = loop.run_in_executor(
            executor,
            lambda: requests.get(
                f"https://api.genius.com/search?q={song} {artist} romaji",
                headers={"Authorization": f"Bearer {GENIUS_API_TOKEN}"},
                timeout=5
            )
        )
    
    # Wait for LRC first
    try:
        lrc_response = await asyncio.wait_for(lrc_future, timeout=5)
        lrc_data = lrc_response.json()
        lrc_text = lrc_data.get("syncedLyrics")
        
        if not lrc_text:
            raise HTTPException(404, "No lyrics found")
        
        # Parse LRC lines
        lrc_lines = parse_lrc_lines(lrc_text)
        
        # Get Genius result if available
        romaji_text = None
        if genius_future:
            try:
                genius_response = await asyncio.wait_for(genius_future, timeout=5)
                genius_data = genius_response.json()
                
                if genius_data['response']['hits']:
                    song_url = genius_data['response']['hits'][0]['result']['url']
                    page = requests.get(song_url, headers={'User-Agent': 'Mozilla/5.0'})
                    soup = BeautifulSoup(page.text, 'html.parser')
                    lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
                    
                    full_text = []
                    for div in lyrics_divs:
                        text = div.get_text(separator='\n', strip=True)
                        full_text.append(text)
                    
                    romaji_text = '\n'.join(full_text)
            except:
                pass  # Genius is optional
        
        # Process alignment
        final_lyrics = []
        source = ""
        
        if romaji_text:
            print("✨ Found Genius Romaji! Aligning...")
            final_lyrics = await smart_align_lyrics(lrc_lines, romaji_text)
            
            if final_lyrics and len(final_lyrics) == len(lrc_lines):
                source = "Genius + Smart Align"
            else:
                # Fallback to translation
                translated = await batch_translate_lines([l['reference'] for l in lrc_lines])
                final_lyrics = [
                    f"{lrc_lines[i]['timestamp']} {translated[i]}" 
                    for i in range(len(lrc_lines))
                ]
                source = "AI Translation (Fallback)"
        else:
            print("🤖 No Genius Romaji found. Translating...")
            translated = await batch_translate_lines([l['reference'] for l in lrc_lines])
            final_lyrics = [
                f"{lrc_lines[i]['timestamp']} {translated[i]}" 
                for i in range(len(lrc_lines))
            ]
            source = "AI Translation"
        
        result = {
            "lyrics": '\n'.join(final_lyrics),
            "song": song,
            "artist": artist,
            "source": source,
            "line_count": len(final_lyrics),
            "processing_time": round(time.time() - start_time, 2)
        }
        
        # Cache
        song_cache[cache_key] = result
        if redis_client: 
            redis_client.setex(cache_key, 86400, json.dumps(result))  # 1 day cache
        
        return result
        
    except asyncio.TimeoutError:
        raise HTTPException(504, "Timeout fetching lyrics")
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, "Processing failed")

# --- 6. STREAMING ENDPOINT (REAL-TIME) ---
@app.get("/stream_song")
async def stream_song(song: str, artist: str):
    """Stream lyrics as they become available (for real-time display)"""
    async def generate():
        # Send initial metadata
        yield json.dumps({"status": "started", "song": song, "artist": artist}) + "\n"
        
        # Get timestamps first (fast)
        lrc_data = await fetch_lrc_timestamps(song, artist)
        if not lrc_data:
            yield json.dumps({"error": "No timestamps found"}) + "\n"
            return
        
        yield json.dumps({
            "status": "timestamps_loaded", 
            "count": len(lrc_data),
            "progress": 0.2
        }) + "\n"
        
        # Start Genius fetch in background
        genius_task = asyncio.create_task(fetch_genius_lyrics_parallel(song, artist))
        
        # Send first few translated lines immediately (for instant display)
        first_lines = lrc_data[:15]
        translated = await batch_translate_lines([l['reference'] for l in first_lines])
        
        for i in range(len(first_lines)):
            yield json.dumps({
                "line": f"{first_lines[i]['timestamp']} {translated[i]}",
                "index": i,
                "total": len(lrc_data),
                "progress": 0.3 + (i / len(lrc_data) * 0.3)
            }) + "\n"
        
        # Wait for Genius
        romaji_text = await genius_task
        
        if romaji_text:
            # Align and send the rest
            yield json.dumps({"status": "genius_loaded", "aligning": True, "progress": 0.6}) + "\n"
            
            aligned = await smart_align_lyrics(lrc_data[15:], romaji_text)
            for i, line in enumerate(aligned, start=15):
                yield json.dumps({
                    "line": line,
                    "index": i,
                    "total": len(lrc_data),
                    "progress": 0.6 + (i / len(lrc_data) * 0.4)
                }) + "\n"
        else:
            # Continue with translation
            remaining = lrc_data[15:]
            translated = await batch_translate_lines([l['reference'] for l in remaining])
            
            for i in range(len(remaining)):
                yield json.dumps({
                    "line": f"{remaining[i]['timestamp']} {translated[i]}",
                    "index": i + 15,
                    "total": len(lrc_data),
                    "progress": 0.6 + ((i + 15) / len(lrc_data) * 0.4)
                }) + "\n"
        
        yield json.dumps({"status": "complete", "progress": 1.0}) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")

# --- 7. EXISTING ENDPOINTS (UPDATED) ---
@app.get("/")
async def root():
    return {"status": "Online", "mode": "Smart Hybrid v2"}

@app.get("/convert")
async def convert_single_line(text: str = ""):
    """Quick single line conversion"""
    if not text: 
        raise HTTPException(400, "No text")
    
    if text in line_cache: 
        return {"original": text, "romaji": line_cache[text]}
    
    if not client: 
        return {"original": text, "romaji": text}
    
    try:
        romaji = await translate_line(text)
        line_cache[text] = romaji
        return {"original": text, "romaji": romaji}
    except: 
        return {"original": text, "romaji": text}

@app.get("/get_song")
async def get_song_endpoint(song: str, artist: str):
    """Main endpoint for song lyrics"""
    return await process_song(song, artist)

@app.delete("/clear_cache")
async def clear():
    """Clear all caches"""
    song_cache.clear()
    line_cache.clear()
    if redis_client: 
        redis_client.flushdb()
    return {"status": "cleared"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    status = {
        "deepseek": bool(client),
        "redis": redis_client.ping() if redis_client else False,
        "genius": bool(GENIUS_API_TOKEN),
        "cache_size": len(song_cache)
    }
    return status

# Quick align fallback (for simple cases)
def quick_align(lrc_lines, romaji_text):
    """Simple word-based alignment as fallback"""
    romaji_lines = [l.strip() for l in romaji_text.split('\n') if l.strip()]
    
    result = []
    romaji_idx = 0
    
    for lrc in lrc_lines:
        jp_words = set(lrc['reference'].split())
        
        # Look for matching line in nearby positions
        best_score = 0
        best_line = None
        
        for i in range(max(0, romaji_idx-2), min(len(romaji_lines), romaji_idx+3)):
            romaji_words = set(romaji_lines[i].split())
            overlap = len(jp_words & romaji_words)
            
            if overlap > best_score:
                best_score = overlap
                best_line = romaji_lines[i]
                romaji_idx = i + 1
        
        if best_score >= 1:  # At least one word overlap
            result.append(f"{lrc['timestamp']} {best_line}")
        else:
            result.append(f"{lrc['timestamp']} {lrc['reference']}")
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
