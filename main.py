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
import jaconv  # Japanese conversion library

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

# --- JAPANESE TEXT PROCESSING WITH JACONV ---
def normalize_japanese(text: str) -> str:
    """Normalize Japanese text for better matching using jaconv"""
    if not text:
        return ""
    
    try:
        # Normalize using jaconv
        text = jaconv.normalize(text)
        # Convert katakana to hiragana for consistency
        text = jaconv.kata2hira(text)
        # Remove punctuation and normalize
        text = unicodedata.normalize('NFKC', text.lower())
        text = re.sub(r'[「」【】『』()\[\]{}、。！？・・・]', '', text)
        return text.strip()
    except Exception as e:
        # Fallback if jaconv fails
        print(f"jaconv error: {e}")
        text = unicodedata.normalize('NFKC', text.lower())
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

def japanese_to_romaji_approximate(japanese: str) -> str:
    """Create an approximate Romaji version for matching with common word mappings"""
    # Common Japanese word to Romaji mappings (for matching purposes)
    conversions = {
        '今': 'ima',
        '現在': 'genzai',
        '私': 'watashi',
        '僕': 'boku',
        '俺': 'ore',
        'あなた': 'anata',
        '君': 'kimi',
        '愛': 'ai',
        '恋': 'koi',
        '心': 'kokoro',
        '言葉': 'kotoba',
        '世界': 'sekai',
        '夢': 'yume',
        '未来': 'mirai',
        '過去': 'kako',
        '時間': 'jikan',
        '場所': 'basho',
        '声': 'koe',
        '夜': 'yoru',
        '日': 'hi',
        '風': 'kaze',
        '雨': 'ame',
        '雪': 'yuki',
        '星': 'hoshi',
        '月': 'tsuki',
        '太陽': 'taiyou',
        '空': 'sora',
        '海': 'umi',
        '山': 'yama',
        '街': 'machi',
        '家': 'ie',
        '手': 'te',
        '目': 'me',
        '涙': 'namida',
        '笑顔': 'egao',
        '幸せ': 'shiawase',
        '悲しい': 'kanashii',
        '嬉しい': 'ureshii',
        '楽しい': 'tanoshii',
        '痛い': 'itai',
        '強い': 'tsuyoi',
        '優しい': 'yasashii',
        '大好き': 'daisuki',
        '好き': 'suki',
        '会う': 'au',
        '行く': 'iku',
        '来る': 'kuru',
        '見る': 'miru',
        '聞く': 'kiku',
        '言う': 'iu',
        '思う': 'omou',
        '感じる': 'kanjiru',
        '信じる': 'shinjiru',
    }
    
    result = japanese
    for jp, romaji in conversions.items():
        if jp in result:
            result = result.replace(jp, romaji)
    
    return result.lower()

def calculate_similarity_with_synonyms(japanese: str, romaji: str) -> Tuple[float, str]:
    """
    Calculate similarity with synonym handling
    Returns: (similarity_score, reason)
    """
    # Normalize both
    norm_jp = normalize_japanese(japanese)
    norm_romaji = romaji.lower().strip()
    
    # Generate approximate romaji from Japanese
    approx_romaji = japanese_to_romaji_approximate(japanese)
    
    # Check for direct matches
    if norm_jp == norm_romaji or approx_romaji == norm_romaji:
        return 1.0, "exact_match"
    
    # Word-based matching
    jp_words = set(norm_jp.split())
    romaji_words = set(norm_romaji.split())
    approx_words = set(approx_romaji.split())
    
    # Calculate various similarity metrics
    direct_word_overlap = len(jp_words & romaji_words) / max(len(jp_words), 1)
    approx_word_overlap = len(approx_words & romaji_words) / max(len(approx_words), 1)
    
    # Sequence matching
    seq_direct = SequenceMatcher(None, norm_jp, norm_romaji).ratio()
    seq_approx = SequenceMatcher(None, approx_romaji, norm_romaji).ratio()
    
    # Use the best match
    best_score = max(
        seq_direct,
        seq_approx,
        direct_word_overlap,
        approx_word_overlap
    )
    
    # Determine reason
    reasons = []
    if seq_direct > 0.7:
        reasons.append(f"direct_seq:{seq_direct:.2f}")
    if seq_approx > 0.7:
        reasons.append(f"approx_seq:{seq_approx:.2f}")
    if direct_word_overlap > 0.3:
        reasons.append(f"direct_words:{direct_word_overlap:.2f}")
    if approx_word_overlap > 0.3:
        reasons.append(f"approx_words:{approx_word_overlap:.2f}")
    
    reason = "|".join(reasons) if reasons else "low_similarity"
    
    return min(best_score, 1.0), reason

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
async def fetch_genius_lyrics(song: str, artist: str) -> Optional[Tuple[str, str]]:
    """Get structured romaji lyrics from Genius"""
    if not GENIUS_API_TOKEN: 
        return None
    
    try:
        headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
        loop = asyncio.get_event_loop()
        
        # Search for song
        search_query = f"{song} {artist}"
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
        
        # Get the best matching result
        best_hit = None
        for hit in data['response']['hits']:
            result = hit['result']
            # Check if it's likely the right song
            if artist.lower() in result['primary_artist']['name'].lower():
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
        
        # Try to find Romaji lyrics
        romaji_text = ""
        
        # Method 1: Look for Romaji containers
        romaji_divs = soup.find_all('div', class_=re.compile(r'.*romaji.*', re.I))
        for div in romaji_divs:
            text = div.get_text(separator='\n', strip=True)
            if text:
                romaji_text += text + "\n\n"
        
        # Method 2: If no Romaji found, try all lyrics containers
        if not romaji_text.strip():
            lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
            for div in lyrics_divs:
                text = div.get_text(separator='\n', strip=True)
                # Check if it looks like Romaji
                latin = len(re.findall(r'[a-zA-Z]', text))
                japanese = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))
                if latin > japanese:  # More Latin than Japanese
                    romaji_text += text + "\n\n"
        
        # Clean up
        romaji_text = re.sub(r'\[.*?\]', '', romaji_text)
        romaji_text = re.sub(r'\(.*?\)', '', romaji_text)
        romaji_text = re.sub(r'\n\s*\n', '\n', romaji_text)
        romaji_text = romaji_text.strip()
        
        if romaji_text and len(romaji_text) > 30:
            return romaji_text, song_url
        else:
            return None
        
    except Exception as e:
        print(f"Genius fetch error: {e}")
        return None

# --- 3. SMART ALIGNMENT WITH SYNONYM HANDLING ---
async def smart_align_with_synonyms(lrc_lines: List[Dict], romaji_text: str) -> List[str]:
    """Align lyrics with synonym awareness"""
    
    # Clean romaji text
    romaji_lines = []
    for line in romaji_text.split('\n'):
        line = line.strip()
        if line and not re.match(r'^[0-9\.]+$', line):
            romaji_lines.append(line)
    
    if not romaji_lines:
        return []
    
    print(f"🔍 Aligning {len(lrc_lines)} Japanese lines with {len(romaji_lines)} Romaji lines")
    
    aligned = []
    romaji_idx = 0
    match_stats = {"high": 0, "medium": 0, "low": 0}
    
    for lrc_idx, lrc_line in enumerate(lrc_lines):
        japanese = lrc_line['reference']
        best_score = 0
        best_match = ""
        best_reason = ""
        
        # Search in a window around current position
        search_start = max(0, romaji_idx - 3)
        search_end = min(len(romaji_lines), romaji_idx + 7)  # Look ahead more
        
        for i in range(search_start, search_end):
            romaji_line = romaji_lines[i]
            score, reason = calculate_similarity_with_synonyms(japanese, romaji_line)
            
            if score > best_score:
                best_score = score
                best_match = romaji_line
                best_reason = reason
                if score > 0.8:  # Good enough match
                    break  # Early exit for good matches
        
        if best_score > 0.65:  # Confidence threshold
            aligned.append(f"{lrc_line['timestamp']} {best_match}")
            romaji_idx = search_start + 1  # Move forward
            
            if best_score > 0.8:
                match_stats["high"] += 1
            elif best_score > 0.65:
                match_stats["medium"] += 1
                
            if best_score < 0.8 and lrc_idx % 10 == 0:
                print(f"   Line {lrc_idx}: {best_score:.2f} ({best_reason})")
        else:
            # No good match, mark for translation
            aligned.append(f"{lrc_line['timestamp']} {japanese}")  # Placeholder
            match_stats["low"] += 1
            if lrc_idx % 10 == 0:
                print(f"   Line {lrc_idx}: No match ({best_score:.2f})")
    
    print(f"📊 Match stats: High={match_stats['high']}, Medium={match_stats['medium']}, Low={match_stats['low']}")
    
    # Use AI to fix low-confidence lines
    if match_stats["low"] > 0 and client:
        print(f"🤖 Fixing {match_stats['low']} low-confidence lines with AI...")
        aligned = await ai_correct_low_confidence(lrc_lines, romaji_lines, aligned)
    
    return aligned

async def ai_correct_low_confidence(lrc_lines: List[Dict], romaji_lines: List[str], aligned: List[str]) -> List[str]:
    """Use AI to fix lines with low confidence matches"""
    
    # Find indices that need fixing (those with Japanese text)
    fix_indices = []
    for i, line in enumerate(aligned):
        if i < len(lrc_lines) and re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line):
            fix_indices.append(i)
    
    if not fix_indices:
        return aligned
    
    # Prepare context for AI
    context = []
    for idx in fix_indices[:20]:  # Limit to first 20 for token management
        japanese = lrc_lines[idx]['reference']
        timestamp = lrc_lines[idx]['timestamp']
        
        # Get surrounding romaji for context
        romaji_context = []
        for i in range(max(0, idx-2), min(len(romaji_lines), idx+3)):
            romaji_context.append(romaji_lines[i])
        
        context.append({
            "index": idx,
            "japanese": japanese,
            "timestamp": timestamp,
            "surrounding_romaji": romaji_context
        })
    
    prompt = f"""Fix these lyric alignments. The Japanese lines didn't match well with the Romaji.

IMPORTANT WORD CHOICES:
- 今 → ALWAYS use "ima" (never "genzai" in songs)
- 現在 → use "genzai" only if explicitly "current time"
- 私 → "watashi" (default)
- 僕 → "boku" (male)
- 俺 → "ore" (casual male)
- あなた → "anata" (default)
- 君 → "kimi"

For each line, choose the BEST Romaji match from surrounding options OR translate if needed.

LINES TO FIX ({len(context)}):
{json.dumps(context, ensure_ascii=False, indent=2)}

Output JSON: {{"fixed": [{{"index": 0, "romaji": "fixed text"}}, ...]}}
Each output line must be: [timestamp] romaji_text"""

    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(completion.choices[0].message.content)
        fixed_data = data.get("fixed", [])
        
        # Apply fixes
        result = aligned.copy()
        for fix in fixed_data:
            idx = fix.get("index")
            romaji = fix.get("romaji", "").strip()
            if 0 <= idx < len(result) and romaji:
                # Ensure timestamp is preserved
                if not romaji.startswith('['):
                    romaji = f"{lrc_lines[idx]['timestamp']} {romaji}"
                result[idx] = romaji
        
        print(f"✅ AI corrected {len(fixed_data)} lines")
        return result
        
    except Exception as e:
        print(f"AI correction failed: {e}")
        return aligned

# --- 4. BATCH TRANSLATION WITH PREFERENCES ---
async def batch_translate_with_preferences(japanese_lines: List[str]) -> List[str]:
    """Translate with word preference rules"""
    if not client or not japanese_lines:
        return japanese_lines
    
    batch_size = 20
    results = []
    
    for i in range(0, len(japanese_lines), batch_size):
        batch = japanese_lines[i:i+batch_size]
        prompt = f"""Translate these Japanese lyrics to Romaji (Hepburn romanization).

CRITICAL WORD PREFERENCES:
- 今 → ALWAYS "ima" (NEVER "genzai" for song lyrics)
- 現在 → "genzai" (only for emphasis on "current")
- 私 → "watashi" (standard)
- 僕 → "boku" (male)
- 俺 → "ore" (casual male)
- あなた → "anata"
- 君 → "kimi"

Maintain poetic flow, line breaks, and use proper long vowels (ō, ū, etc.).

LINES ({len(batch)}):
{json.dumps(batch, ensure_ascii=False)}

Output JSON: {{"translations": ["romaji1", "romaji2", ...]}}"""
        
        try:
            completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=DEEPSEEK_MODEL,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content)
            translations = data.get("translations", [])
            
            if len(translations) == len(batch):
                results.extend(translations)
            else:
                # Fallback
                for line in batch:
                    trans = await translate_single_line(line)
                    results.append(trans)
        except Exception as e:
            print(f"Translation error: {e}")
            results.extend(batch)
    
    return results

async def translate_single_line(text: str) -> str:
    """Translate single line with preferences"""
    prompt = f"""Translate to Romaji with these rules:
    - 今 → "ima" (never "genzai")
    - 現在 → "genzai" (only if explicitly "current time")
    - Keep natural and poetic
    
    Japanese: {text}
    Romaji:"""
    
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.1,
            max_tokens=100
        )
        return completion.choices[0].message.content.strip()
    except:
        return text

# --- 5. MAIN PROCESSING ---
async def process_song(song: str, artist: str):
    """Main processing function"""
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

    print(f"🚀 Processing: {song} by {artist}")
    start_time = time.time()
    
    try:
        # Get LRC timestamps
        lrc_lines = await fetch_lrc_timestamps(song, artist)
        if not lrc_lines:
            raise HTTPException(404, "No lyrics found")
        
        print(f"📊 Found {len(lrc_lines)} timed lines")
        
        # Try Genius
        genius_result = await fetch_genius_lyrics(song, artist)
        
        final_lyrics = []
        source = ""
        
        if genius_result:
            romaji_text, _ = genius_result
            print("✨ Found Genius Romaji, aligning...")
            
            # Align with synonym handling
            aligned = await smart_align_with_synonyms(lrc_lines, romaji_text)
            
            # Check quality
            japanese_count = sum(1 for line in aligned if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line))
            quality = 1 - (japanese_count / len(aligned)) if aligned else 0
            
            if quality > 0.85 and len(aligned) == len(lrc_lines):
                final_lyrics = aligned
                source = "Genius + Smart Align"
                print(f"✅ Alignment quality: {quality:.1%}")
            else:
                print(f"⚠️ Alignment poor ({quality:.1%}), using AI translation...")
                translated = await batch_translate_with_preferences([l['reference'] for l in lrc_lines])
                final_lyrics = [
                    f"{lrc_lines[i]['timestamp']} {translated[i]}"
                    for i in range(len(lrc_lines))
                ]
                source = "AI Translation (Fallback)"
        else:
            print("🤖 No Genius found, using AI translation...")
            translated = await batch_translate_with_preferences([l['reference'] for l in lrc_lines])
            final_lyrics = [
                f"{lrc_lines[i]['timestamp']} {translated[i]}"
                for i in range(len(lrc_lines))
            ]
            source = "AI Translation"
        
        # Ensure correct line count
        if len(final_lyrics) != len(lrc_lines):
            final_lyrics = final_lyrics[:len(lrc_lines)]
            while len(final_lyrics) < len(lrc_lines):
                idx = len(final_lyrics)
                final_lyrics.append(f"{lrc_lines[idx]['timestamp']} {lrc_lines[idx]['reference']}")
        
        result = {
            "lyrics": '\n'.join(final_lyrics),
            "song": song,
            "artist": artist,
            "source": source,
            "line_count": len(final_lyrics),
            "processing_time": round(time.time() - start_time, 2),
            "success": True
        }
        
        # Cache
        song_cache[cache_key] = result
        if redis_client:
            redis_client.setex(cache_key, 604800, json.dumps(result))
        
        print(f"✅ Completed in {result['processing_time']}s")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

# --- 6. ENDPOINTS ---
@app.get("/")
async def root():
    return {
        "status": "Online", 
        "mode": "Lyrics Processor with Synonym Handling",
        "features": ["Genius Romaji", "AI Translation", "Smart Alignment", "Synonym Awareness"]
    }

@app.get("/convert")
async def convert_single_line(text: str = ""):
    """Quick conversion with preferences"""
    if not text:
        raise HTTPException(400, "No text")
    
    cache_key = f"conv:{hashlib.md5(text.encode()).hexdigest()}"
    if cache_key in line_cache:
        return {"original": text, "romaji": line_cache[cache_key]}
    
    if not client:
        return {"original": text, "romaji": text}
    
    try:
        romaji = await translate_single_line(text)
        line_cache[cache_key] = romaji
        return {"original": text, "romaji": romaji}
    except:
        return {"original": text, "romaji": text}

@app.get("/get_song")
async def get_song_endpoint(song: str, artist: str):
    """Main endpoint"""
    return await process_song(song, artist)

@app.get("/stream_song")
async def stream_song(song: str, artist: str):
    """Streaming endpoint"""
    async def generate():
        yield json.dumps({"status": "started", "song": song, "artist": artist}) + "\n"
        
        result = await process_song(song, artist)
        lyrics = result.get("lyrics", "").split('\n')
        
        for i, line in enumerate(lyrics):
            yield json.dumps({
                "line": line,
                "index": i,
                "total": len(lyrics),
                "progress": (i + 1) / len(lyrics)
            }) + "\n"
        
        yield json.dumps({"status": "complete", "source": result.get("source")}) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.get("/debug_match")
async def debug_match(japanese: str, romaji: str):
    """Debug endpoint to see matching details"""
    score, reason = calculate_similarity_with_synonyms(japanese, romaji)
    approx = japanese_to_romaji_approximate(japanese)
    
    return {
        "japanese": japanese,
        "romaji": romaji,
        "approximate_romaji": approx,
        "similarity_score": round(score, 3),
        "reason": reason,
        "normalized_jp": normalize_japanese(japanese),
        "normalized_romaji": romaji.lower()
    }

@app.delete("/clear_cache")
async def clear_cache():
    """Clear caches"""
    song_cache.clear()
    line_cache.clear()
    if redis_client:
        redis_client.flushdb()
    return {"status": "Cache cleared"}

@app.get("/health")
async def health():
    """Health check"""
    return {
        "deepseek": bool(client),
        "redis": redis_client.ping() if redis_client else False,
        "genius": bool(GENIUS_API_TOKEN),
        "cache_size": len(song_cache)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
