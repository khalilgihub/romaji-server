from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
import requests
import os
import re
import hashlib
import unicodedata
from typing import List, Optional, Dict, Tuple
import json
import redis
from bs4 import BeautifulSoup
import asyncio
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
        except Exception as e:
            print(f"❌ DeepSeek AI Failed: {e}")
    if REDIS_URL:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            redis_client.ping()
            print("✅ Redis Online")
        except Exception as e:
            print(f"❌ Redis Failed: {e}")
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

# --- 1. FETCH TIMESTAMPS (LRCLib) - OPTIMIZED ---
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
    """Fetch LRC timestamps with better error handling"""
    try:
        url = "https://lrclib.net/api/get"
        params = {
            "track_name": song, 
            "artist_name": artist,
            "duration": ""  # Helps with matching
        }
        
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, 
            lambda: requests.get(url, params=params, timeout=5)
        )
        
        if resp.status_code == 200:
            data = resp.json()
            lrc_text = data.get("syncedLyrics")
            if not lrc_text:
                # Try unsynchronized lyrics as fallback
                lrc_text = data.get("plainLyrics")
                if lrc_text:
                    # Create artificial timestamps for unsynced lyrics
                    lines = []
                    for i, line in enumerate(lrc_text.split('\n')):
                        if line.strip():
                            # Create timestamp based on line number (approx 3 sec per line)
                            minutes = (i * 3) // 60
                            seconds = (i * 3) % 60
                            timestamp = f"[{minutes:02d}:{seconds:02d}.00]"
                            lines.append({
                                'timestamp': timestamp,
                                'reference': line.strip()
                            })
                    return lines
                return None
            
            return parse_lrc_lines(lrc_text)
        return None
    except Exception as e:
        print(f"LRC fetch error: {e}")
        return None

# --- 2. IMPROVED GENIUS FETCHER ---
async def fetch_genius_lyrics(song: str, artist: str) -> Optional[Tuple[str, str]]:
    """Get structured romaji lyrics from Genius, returns (lyrics, url)"""
    if not GENIUS_API_TOKEN: 
        return None
    
    try:
        headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
        
        # Search for song
        search_query = f"{song} {artist}"
        loop = asyncio.get_event_loop()
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
        
        # Try to find the best match
        best_hit = None
        for hit in data['response']['hits']:
            result = hit['result']
            # Check if artist matches reasonably well
            if artist.lower() in result['primary_artist']['name'].lower() or \
               result['primary_artist']['name'].lower() in artist.lower():
                best_hit = result
                break
        
        if not best_hit:
            best_hit = data['response']['hits'][0]['result']
        
        song_url = best_hit['url']
        
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
        
        # Look for Romaji specifically
        romaji_sections = []
        
        # Method 1: Try to find Romaji sections
        for div in soup.find_all('div', class_=re.compile(r'.*romaji.*', re.I)):
            if 'romaji' in div.get('class', ''):
                text = div.get_text(separator='\n', strip=True)
                romaji_sections.append(text)
        
        # Method 2: Check all lyrics containers
        if not romaji_sections:
            lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
            for div in lyrics_divs:
                text = div.get_text(separator='\n', strip=True)
                # Check if this looks like Romaji (mostly Latin characters)
                jp_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))
                latin_chars = len(re.findall(r'[a-zA-Z]', text))
                
                if latin_chars > jp_chars * 2:  # Mostly Latin
                    romaji_sections.append(text)
        
        # Method 3: If still no Romaji, take the first lyrics container
        if not romaji_sections and lyrics_divs:
            text = lyrics_divs[0].get_text(separator='\n', strip=True)
            romaji_sections.append(text)
        
        if not romaji_sections:
            return None
        
        romaji_text = '\n\n'.join(romaji_sections)
        
        # Clean up the text
        romaji_text = re.sub(r'\[.*?\]', '', romaji_text)  # Remove bracketed sections
        romaji_text = re.sub(r'\n\s*\n', '\n', romaji_text)  # Remove empty lines
        romaji_text = romaji_text.strip()
        
        return romaji_text, song_url
        
    except Exception as e:
        print(f"Genius fetch error: {e}")
        return None

# --- 3. ADVANCED MATCHING ALGORITHM ---
def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using multiple metrics"""
    # Normalize texts
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    # 1. Sequence matcher ratio
    seq_ratio = SequenceMatcher(None, norm1, norm2).ratio()
    
    # 2. Word overlap ratio
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    if words1 or words2:
        word_ratio = len(words1 & words2) / max(len(words1), len(words2))
    else:
        word_ratio = 0
    
    # 3. Character n-gram similarity (for partial matches)
    def get_ngrams(text, n=3):
        return [text[i:i+n] for i in range(len(text)-n+1)]
    
    ngrams1 = set(get_ngrams(norm1))
    ngrams2 = set(get_ngrams(norm2))
    if ngrams1 or ngrams2:
        ngram_ratio = len(ngrams1 & ngrams2) / max(len(ngrams1), len(ngrams2))
    else:
        ngram_ratio = 0
    
    # Weighted combination
    return 0.4 * seq_ratio + 0.4 * word_ratio + 0.2 * ngram_ratio

async def advanced_align_lyrics(lrc_lines: List[Dict], romaji_text: str) -> List[str]:
    """Advanced alignment using multiple strategies"""
    
    # Clean and split romaji text
    romaji_lines = []
    for line in romaji_text.split('\n'):
        line = line.strip()
        if line and not re.match(r'^[0-9\.]+$', line):  # Skip timestamps and numbers
            romaji_lines.append(line)
    
    if not romaji_lines:
        return []
    
    # Strategy 1: Try to match line by line with dynamic window
    aligned = []
    romaji_idx = 0
    max_skip = 3  # Allow skipping up to 3 romaji lines
    
    for lrc_idx, lrc_line in enumerate(lrc_lines):
        japanese = lrc_line['reference']
        best_score = 0
        best_match_idx = -1
        best_match_text = ""
        
        # Search in a window around current position
        search_start = max(0, romaji_idx - max_skip)
        search_end = min(len(romaji_lines), romaji_idx + max_skip + 1)
        
        for i in range(search_start, search_end):
            similarity = calculate_similarity(japanese, romaji_lines[i])
            if similarity > best_score and similarity > 0.3:
                best_score = similarity
                best_match_idx = i
                best_match_text = romaji_lines[i]
        
        if best_match_idx >= 0:
            aligned.append(f"{lrc_line['timestamp']} {best_match_text}")
            romaji_idx = best_match_idx + 1
        else:
            # No good match found, use current line and advance
            if romaji_idx < len(romaji_lines):
                aligned.append(f"{lrc_line['timestamp']} {romaji_lines[romaji_idx]}")
                romaji_idx += 1
            else:
                # Out of romaji lines, use Japanese
                aligned.append(f"{lrc_line['timestamp']} {japanese}")
    
    # Strategy 2: If alignment seems poor, try AI alignment
    japanese_count = sum(1 for line in aligned if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line))
    if japanese_count > len(aligned) * 0.3:  # More than 30% Japanese
        print("⚠️ Poor alignment detected, trying AI correction...")
        try:
            ai_aligned = await ai_correct_alignment(lrc_lines, romaji_text, aligned)
            if ai_aligned:
                return ai_aligned
        except:
            pass
    
    return aligned

async def ai_correct_alignment(lrc_lines: List[Dict], romaji_text: str, current_alignment: List[str]) -> Optional[List[str]]:
    """Use AI to correct poor alignment"""
    if not client:
        return None
    
    prompt = f"""Fix this lyrics alignment. I have {len(lrc_lines)} Japanese lines with timestamps, 
    and Romaji lyrics. Current alignment has issues. Please provide better alignment.

    JAPANESE LINES WITH TIMESTAMPS:
    {json.dumps([f"{l['timestamp']} {l['reference']}" for l in lrc_lines[:30]], ensure_ascii=False)}

    ROMAJI LYRICS:
    {romaji_text[:2000]}

    CURRENT (FLAWED) ALIGNMENT:
    {json.dumps(current_alignment[:30], ensure_ascii=False)}

    RULES:
    1. Output EXACTLY {len(lrc_lines)} lines
    2. Each line must start with the EXACT timestamp from Japanese lines
    3. Map Romaji to Japanese as accurately as possible
    4. If unsure, translate Japanese to Romaji
    5. Output format: ["[00:00.00] Romaji text", ...]

    OUTPUT JSON: {{"aligned": ["line1", "line2", ...]}}"""
    
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        data = json.loads(completion.choices[0].message.content)
        return data.get("aligned", [])
    except:
        return None

# --- 4. BATCH TRANSLATION (OPTIMIZED) ---
async def batch_translate_lines(japanese_lines: List[str]) -> List[str]:
    """Translate multiple lines at once for efficiency"""
    if not client or not japanese_lines:
        return japanese_lines
    
    # Group lines for efficient processing
    batch_size = 30
    results = []
    
    for i in range(0, len(japanese_lines), batch_size):
        batch = japanese_lines[i:i+batch_size]
        prompt = f"""Translate these Japanese lyrics to Romaji (Hepburn romanization).
        
        IMPORTANT:
        1. Preserve the original meaning and poetic flow
        2. Maintain line breaks exactly (output same number of lines)
        3. Use proper romanization with long vowels (ō, ū, etc.)
        4. Keep song structure (verse, chorus markers if present)
        
        LINES TO TRANSLATE ({len(batch)} lines):
        {json.dumps(batch, ensure_ascii=False)}
        
        Output JSON: {{"translations": ["romaji line 1", "romaji line 2", ...]}}"""
        
        try:
            completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=DEEPSEEK_MODEL,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content)
            translations = data.get("translations", [])
            
            # Ensure we have the right number of translations
            if len(translations) == len(batch):
                results.extend(translations)
            else:
                # Fallback: translate individually
                for line in batch:
                    try:
                        trans = await translate_line_simple(line)
                        results.append(trans)
                    except:
                        results.append(line)
        except Exception as e:
            print(f"Batch translation error: {e}")
            # Fallback to original text
            results.extend(batch)
    
    return results

async def translate_line_simple(text: str) -> str:
    """Simple single line translation"""
    prompt = f"Translate to Romaji (Hepburn): {text}"
    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=DEEPSEEK_MODEL,
        temperature=0.1,
        max_tokens=100
    )
    return completion.choices[0].message.content.strip()

# --- 5. MAIN PROCESSING (COMPLETE REWORK) ---
async def process_song(song: str, artist: str):
    """Main processing pipeline with fallback strategies"""
    cache_key = f"song:{hashlib.md5(f'{song.lower()}:{artist.lower()}'.encode()).hexdigest()}"
    
    # Check cache
    if cache_key in song_cache:
        return song_cache[cache_key]
    
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached:
            result = json.loads(cached)
            song_cache[cache_key] = result
            return result

    print(f"🚀 Processing: {song} by {artist}...")
    start_time = time.time()
    
    try:
        # STEP 1: Fetch LRC timestamps (primary source)
        lrc_lines = await fetch_lrc_timestamps(song, artist)
        if not lrc_lines:
            raise HTTPException(404, "No lyrics found on LRCLib")
        
        print(f"📊 Found {len(lrc_lines)} timed lines")
        
        # STEP 2: Try to fetch Genius Romaji
        genius_result = await fetch_genius_lyrics(song, artist)
        romaji_text = None
        
        if genius_result:
            romaji_text, genius_url = genius_result
            print(f"🎵 Found Genius Romaji ({len(romaji_text)} chars)")
        
        # STEP 3: Process alignment
        final_lyrics = []
        source = ""
        processing_details = []
        
        if romaji_text:
            # Strategy A: Advanced alignment
            aligned = await advanced_align_lyrics(lrc_lines, romaji_text)
            
            # Check alignment quality
            japanese_count = sum(1 for line in aligned if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line))
            alignment_ratio = 1 - (japanese_count / len(aligned)) if aligned else 0
            
            if alignment_ratio > 0.7 and len(aligned) == len(lrc_lines):
                final_lyrics = aligned
                source = "Genius + Advanced Align"
                processing_details.append(f"Alignment quality: {alignment_ratio:.1%}")
            else:
                # Strategy B: Translate everything
                print("🔄 Alignment poor, translating all lines...")
                translated = await batch_translate_lines([l['reference'] for l in lrc_lines])
                final_lyrics = [
                    f"{lrc_lines[i]['timestamp']} {translated[i]}" 
                    for i in range(len(lrc_lines))
                ]
                source = "AI Translation (Fallback)"
                processing_details.append(f"Alignment poor ({alignment_ratio:.1%}), used translation")
        else:
            # No Genius, translate everything
            print("🔤 No Genius Romaji found, translating...")
            translated = await batch_translate_lines([l['reference'] for l in lrc_lines])
            final_lyrics = [
                f"{lrc_lines[i]['timestamp']} {translated[i]}" 
                for i in range(len(lrc_lines))
            ]
            source = "AI Translation"
        
        # Ensure we have the right number of lines
        if len(final_lyrics) != len(lrc_lines):
            print(f"⚠️ Line count mismatch ({len(final_lyrics)} vs {len(lrc_lines)}), adjusting...")
            # Pad or truncate to match
            if len(final_lyrics) < len(lrc_lines):
                for i in range(len(final_lyrics), len(lrc_lines)):
                    final_lyrics.append(f"{lrc_lines[i]['timestamp']} {lrc_lines[i]['reference']}")
            else:
                final_lyrics = final_lyrics[:len(lrc_lines)]
        
        # Create result
        result = {
            "lyrics": '\n'.join(final_lyrics),
            "song": song,
            "artist": artist,
            "source": source,
            "line_count": len(final_lyrics),
            "processing_time": round(time.time() - start_time, 2),
            "details": processing_details,
            "cached": False
        }
        
        # Cache the result
        song_cache[cache_key] = result
        if redis_client:
            redis_client.setex(cache_key, 604800, json.dumps(result))  # 7 days cache
        
        print(f"✅ Completed in {result['processing_time']}s via {source}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Processing error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

# --- 6. STREAMING ENDPOINT (REAL-TIME) ---
@app.get("/stream_song")
async def stream_song(song: str, artist: str):
    """Stream lyrics as they become available"""
    async def generate():
        cache_key = f"stream:{hashlib.md5(f'{song.lower()}:{artist.lower()}'.encode()).hexdigest()}"
        
        # Check cache first
        cached_result = None
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                cached_result = json.loads(cached)
        
        if cached_result:
            # Send cached result quickly
            yield json.dumps({
                "status": "cached",
                "song": song,
                "artist": artist,
                "progress": 1.0
            }) + "\n"
            
            lines = cached_result.get("lyrics", "").split('\n')
            for i, line in enumerate(lines):
                yield json.dumps({
                    "line": line,
                    "index": i,
                    "total": len(lines),
                    "progress": (i + 1) / len(lines)
                }) + "\n"
            
            yield json.dumps({
                "status": "complete",
                "source": cached_result.get("source", "cache"),
                "cached": True
            }) + "\n"
            return
        
        # Not cached, process in real-time
        yield json.dumps({
            "status": "started", 
            "song": song, 
            "artist": artist,
            "progress": 0.1
        }) + "\n"
        
        # Fetch LRC first
        lrc_lines = await fetch_lrc_timestamps(song, artist)
        if not lrc_lines:
            yield json.dumps({"error": "No lyrics found"}) + "\n"
            return
        
        yield json.dumps({
            "status": "timestamps_loaded",
            "count": len(lrc_lines),
            "progress": 0.3
        }) + "\n"
        
        # Start Genius fetch in background
        genius_task = asyncio.create_task(fetch_genius_lyrics(song, artist))
        
        # Send first batch immediately
        first_batch = min(10, len(lrc_lines))
        if first_batch > 0:
            first_lines = lrc_lines[:first_batch]
            translated = await batch_translate_lines([l['reference'] for l in first_lines])
            
            for i in range(first_batch):
                yield json.dumps({
                    "line": f"{first_lines[i]['timestamp']} {translated[i]}",
                    "index": i,
                    "total": len(lrc_lines),
                    "progress": 0.4 + (i / len(lrc_lines) * 0.2)
                }) + "\n"
        
        # Get Genius result
        genius_result = await genius_task
        romaji_text = None
        
        if genius_result:
            romaji_text, _ = genius_result
            yield json.dumps({
                "status": "genius_loaded",
                "progress": 0.6
            }) + "\n"
            
            # Process remaining lines with alignment
            remaining = lrc_lines[first_batch:]
            if romaji_text:
                aligned = await advanced_align_lyrics(remaining, romaji_text)
                for i, line in enumerate(aligned, start=first_batch):
                    yield json.dumps({
                        "line": line,
                        "index": i,
                        "total": len(lrc_lines),
                        "progress": 0.6 + (i / len(lrc_lines) * 0.4)
                    }) + "\n"
            else:
                # No genius, translate remaining
                translated = await batch_translate_lines([l['reference'] for l in remaining])
                for i in range(len(remaining)):
                    yield json.dumps({
                        "line": f"{remaining[i]['timestamp']} {translated[i]}",
                        "index": i + first_batch,
                        "total": len(lrc_lines),
                        "progress": 0.6 + ((i + first_batch) / len(lrc_lines) * 0.4)
                    }) + "\n"
        else:
            # No genius at all, translate everything
            remaining = lrc_lines[first_batch:]
            translated = await batch_translate_lines([l['reference'] for l in remaining])
            for i in range(len(remaining)):
                yield json.dumps({
                    "line": f"{remaining[i]['timestamp']} {translated[i]}",
                    "index": i + first_batch,
                    "total": len(lrc_lines),
                    "progress": 0.6 + ((i + first_batch) / len(lrc_lines) * 0.4)
                }) + "\n"
        
        yield json.dumps({
            "status": "complete",
            "progress": 1.0
        }) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")

# --- 7. ENDPOINTS ---
@app.get("/")
async def root():
    return {
        "status": "Online", 
        "mode": "Enhanced Lyrics Processor",
        "features": ["Genius Romaji", "AI Translation", "Smart Alignment", "Real-time Streaming"]
    }

@app.get("/convert")
async def convert_single_line(text: str = ""):
    """Quick single line conversion"""
    if not text: 
        raise HTTPException(400, "No text provided")
    
    cache_key = f"convert:{hashlib.md5(text.encode()).hexdigest()}"
    if cache_key in line_cache:
        return {"original": text, "romaji": line_cache[cache_key]}
    
    if not client:
        return {"original": text, "romaji": text}
    
    try:
        romaji = await translate_line_simple(text)
        line_cache[cache_key] = romaji
        return {"original": text, "romaji": romaji}
    except Exception as e:
        print(f"Conversion error: {e}")
        return {"original": text, "romaji": text}

@app.get("/get_song")
async def get_song_endpoint(song: str, artist: str):
    """Main endpoint for song lyrics"""
    return await process_song(song, artist)

@app.get("/search")
async def search_songs(query: str):
    """Search for songs on LRCLib"""
    try:
        url = "https://lrclib.net/api/search"
        params = {"q": query}
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        
        # Format results
        results = []
        for item in data.get("data", [])[:10]:  # Limit to 10 results
            results.append({
                "song": item.get("trackName"),
                "artist": item.get("artistName"),
                "album": item.get("albumName"),
                "duration": item.get("duration"),
                "has_lyrics": bool(item.get("syncedLyrics"))
            })
        
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(500, f"Search failed: {e}")

@app.delete("/clear_cache")
async def clear_cache():
    """Clear all caches"""
    song_cache.clear()
    line_cache.clear()
    if redis_client:
        redis_client.flushdb()
    return {"status": "Cache cleared", "memory_items": len(song_cache)}

@app.get("/health")
async def health():
    """Health check endpoint"""
    status = {
        "deepseek": bool(client),
        "redis": redis_client.ping() if redis_client else False,
        "genius": bool(GENIUS_API_TOKEN),
        "cache": {
            "songs": len(song_cache),
            "lines": len(line_cache)
        }
    }
    return status

@app.get("/stats")
async def stats():
    """Get usage statistics"""
    return {
        "cache_size": len(song_cache),
        "line_cache_size": len(line_cache),
        "thread_pool_workers": executor._max_workers,
        "active_tasks": len(asyncio.all_tasks())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
