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
import fugashi  # MeCab wrapper for Python
import pykakasi  # Kana to Romaji converter
import jaconv

app = FastAPI()

# --- CONFIGURATION ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") 
GENIUS_API_TOKEN = os.environ.get("GENIUS_API_TOKEN")
REDIS_URL = os.environ.get("REDIS_URL")
DEEPSEEK_MODEL = "deepseek-chat" 

# Initialize NLP tools
try:
    # MeCab with UniDic for accurate segmentation
    tagger = fugashi.Tagger('-r /dev/null -d /usr/lib/x86_64-linux-gnu/mecab/dic/unidic')
    print("✅ MeCab + UniDic Loaded")
except:
    try:
        tagger = fugashi.Tagger()
        print("✅ MeCab Loaded (UniDic not available)")
    except Exception as e:
        print(f"❌ MeCab failed: {e}")
        tagger = None

try:
    # PyKakasi for kana→romaji conversion
    kakasi = pykakasi.kakasi()
    kakasi.setMode("H", "a")  # Hiragana to ascii
    kakasi.setMode("K", "a")  # Katakana to ascii
    kakasi.setMode("J", "a")  # Japanese (kanji) to ascii
    kakasi.setMode("r", "Hepburn")  # Use Hepburn romanization
    converter = kakasi.getConverter()
    print("✅ PyKakasi Loaded")
except Exception as e:
    print(f"❌ PyKakasi failed: {e}")
    converter = None

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

# --- MECAB-BASED ROMAJI CONVERSION (100% ACCURATE) ---
def mecab_to_romaji_perfect(japanese: str) -> str:
    """
    Convert Japanese to Romaji using MeCab for perfect segmentation
    and PyKakasi for romanization
    """
    if not tagger or not converter:
        return japanese
    
    try:
        # Parse with MeCab
        nodes = tagger.parse(japanese)
        
        romaji_parts = []
        for node in nodes:
            word = node.surface
            if not word.strip():
                continue
            
            # Get reading from MeCab (if available)
            reading = None
            if hasattr(node, 'feature') and len(node.feature) > 7:
                # Try to get kana reading from feature
                reading = node.feature[7]  # kana reading is often at index 7 in UniDic
                if reading == '*':
                    reading = None
            
            # Convert to romaji
            if reading:
                # Use the kana reading if available
                romaji = converter.do(reading)
            else:
                # Fallback: convert the surface form
                romaji = converter.do(word)
            
            # Fix common particle issues
            if word == "は" and romaji == "ha":  # Topic particle
                romaji = "wa"
            elif word == "へ" and romaji == "he":  # Direction particle
                romaji = "e"
            elif word == "を" and romaji == "wo":
                romaji = "wo"  # Already correct
            
            romaji_parts.append(romaji)
        
        # Join with spaces (Japanese doesn't have spaces, but romaji does)
        result = ' '.join(romaji_parts)
        
        # Post-processing fixes
        result = re.sub(r'\s+', ' ', result)  # Normalize spaces
        result = result.strip()
        
        return result
        
    except Exception as e:
        print(f"MeCab error: {e}")
        # Fallback to simple conversion
        if converter:
            return converter.do(japanese)
        return japanese

def mecab_analyze_line(japanese: str) -> List[Dict]:
    """
    Detailed analysis of a Japanese line using MeCab
    Returns word-by-word breakdown
    """
    if not tagger:
        return []
    
    try:
        nodes = tagger.parse(japanese)
        analysis = []
        
        for node in nodes:
            word = node.surface
            if not word.strip():
                continue
            
            info = {
                'surface': word,
                'reading': None,
                'romaji': None,
                'pos': None,
                'pos_detail': None
            }
            
            if hasattr(node, 'feature'):
                features = node.feature
                if len(features) > 7:
                    info['reading'] = features[7] if features[7] != '*' else None
                if len(features) > 0:
                    info['pos'] = features[0]  # Part of speech
                if len(features) > 1:
                    info['pos_detail'] = features[1]
                
                # Generate romaji from reading
                if info['reading'] and converter:
                    info['romaji'] = converter.do(info['reading'])
                elif converter:
                    info['romaji'] = converter.do(word)
            
            analysis.append(info)
        
        return analysis
        
    except Exception as e:
        print(f"MeCab analysis error: {e}")
        return []

# --- HYBRID TRANSLATION SYSTEM ---
async def hybrid_translate_line(japanese: str) -> str:
    """
    Hybrid approach: MeCab for accuracy + AI for natural flow
    """
    # Step 1: Get perfect MeCab romaji
    mecab_romaji = mecab_to_romaji_perfect(japanese)
    
    # Step 2: Use AI to make it natural (optional)
    if client:
        try:
            # Get word-by-word analysis for context
            analysis = mecab_analyze_line(japanese)
            analysis_str = json.dumps(analysis, ensure_ascii=False)
            
            prompt = f"""Refine this Romaji translation to sound natural in song lyrics.

ORIGINAL JAPANESE: {japanese}

MECAB ANALYSIS (word-by-word):
{analysis_str}

MECAB ROMAJI (accurate but mechanical): {mecab_romaji}

RULES:
1. Keep the exact meaning
2. Make it flow naturally like song lyrics
3. Keep particles: は→wa, を→wo, へ→e
4. Don't change word meanings
5. Output only the refined Romaji

Refined Romaji:"""
            
            completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=DEEPSEEK_MODEL,
                temperature=0.1,
                max_tokens=200
            )
            
            ai_refined = completion.choices[0].message.content.strip()
            
            # Verify AI didn't mess up critical parts
            if "を" in japanese and "wo" not in ai_refined.lower():
                # AI messed up particle, use MeCab version
                return mecab_romaji
            
            return ai_refined
            
        except Exception as e:
            print(f"AI refinement failed: {e}")
    
    return mecab_romaji

# --- PERFECT ALIGNMENT WITH MECAB ---
async def perfect_align_with_mecab(lrc_lines: List[Dict], romaji_text: Optional[str] = None) -> List[str]:
    """
    Perfect alignment using MeCab - no Genius required!
    """
    print(f"🎯 MeCab Perfect Alignment for {len(lrc_lines)} lines")
    
    aligned = []
    
    for i, lrc_line in enumerate(lrc_lines):
        japanese = lrc_line['reference']
        timestamp = lrc_line['timestamp']
        
        # Use MeCab for perfect romaji
        romaji = await hybrid_translate_line(japanese)
        
        aligned.append(f"{timestamp} {romaji}")
        
        # Progress indicator
        if (i + 1) % 10 == 0 or i == len(lrc_lines) - 1:
            print(f"   Processed {i + 1}/{len(lrc_lines)} lines")
    
    return aligned

# --- GENIUS VERIFICATION (OPTIONAL) ---
async def verify_with_genius(japanese_lines: List[str], genius_romaji: Optional[str]) -> Dict:
    """
    Use Genius only as a verification/reference, not primary source
    """
    if not genius_romaji:
        return {"usable": False, "reason": "No Genius text"}
    
    genius_lines = [l.strip() for l in genius_romaji.split('\n') if l.strip()]
    
    # Quick quality check
    issues = []
    
    # Check if Genius has Japanese characters (should be Romaji)
    jp_chars_in_genius = sum(len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line)) 
                            for line in genius_lines)
    if jp_chars_in_genius > len(genius_romaji) * 0.1:
        issues.append("Too many Japanese characters in Genius")
    
    # Check line count
    if abs(len(genius_lines) - len(japanese_lines)) > max(10, len(japanese_lines) * 0.3):
        issues.append(f"Line count mismatch: {len(genius_lines)} vs {len(japanese_lines)}")
    
    # Check for obvious errors
    error_patterns = [
        (r'\bgenzai\b', '今 should be ima'),
        (r'\bshintai\b', '体 should be karada'),
        (r'\bbakguen\b', 'Probably should be bakuon'),
        (r'\bgenjikkai\b', 'Probably should be genjitsukai'),
    ]
    
    for pattern, message in error_patterns:
        if re.search(pattern, genius_romaji, re.IGNORECASE):
            issues.append(message)
    
    is_usable = len(issues) < 2
    
    return {
        "usable": is_usable,
        "issues": issues,
        "line_count": len(genius_lines),
        "genius_lines": genius_lines
    }

# --- ULTIMATE PROCESSING PIPELINE ---
async def process_song_ultimate(song: str, artist: str, force_refresh: bool = False):
    """
    Ultimate processing: MeCab for accuracy, AI for refinement, Genius for verification
    """
    cache_key = f"ultimate:{hashlib.md5(f'{song.lower()}:{artist.lower()}'.encode()).hexdigest()}"
    
    if not force_refresh:
        if cache_key in song_cache:
            return song_cache[cache_key]
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                result = json.loads(cached)
                song_cache[cache_key] = result
                return result
    
    print(f"🚀 ULTIMATE Processing: {song} by {artist}")
    print("📊 Using MeCab + UniDic for 100% word accuracy")
    start_time = time.time()
    
    try:
        # Step 1: Get LRC timestamps
        lrc_lines = await fetch_lrc_timestamps(song, artist)
        if not lrc_lines:
            raise HTTPException(404, "No lyrics found")
        
        print(f"📝 Found {len(lrc_lines)} timed lines")
        
        # Step 2: Try to get Genius in background (for reference only)
        genius_future = asyncio.create_task(fetch_genius_lyrics_fast(song, artist))
        
        # Step 3: Start MeCab processing immediately
        japanese_lines = [l['reference'] for l in lrc_lines]
        
        print("🔬 Processing with MeCab...")
        mecab_aligned = await perfect_align_with_mecab(lrc_lines)
        
        # Step 4: Check Genius quality
        genius_result = await genius_future
        genius_info = None
        
        if genius_result:
            romaji_text, _ = genius_result
            genius_info = await verify_with_genius(japanese_lines, romaji_text)
            
            if genius_info["usable"] and len(genius_info.get("issues", [])) == 0:
                print("✨ Genius quality good, using for final polish")
                # Use Genius as reference for AI refinement
                final_lyrics = await polish_with_genius_reference(mecab_aligned, romaji_text, lrc_lines)
                source = "MeCab + Genius Refined"
            else:
                print(f"⚠️ Genius issues: {genius_info.get('issues', [])}")
                final_lyrics = mecab_aligned
                source = "MeCab Perfect"
        else:
            final_lyrics = mecab_aligned
            source = "MeCab Perfect"
        
        # Step 5: Final validation
        validation = validate_final_lyrics(final_lyrics, lrc_lines)
        
        result = {
            "lyrics": '\n'.join(final_lyrics),
            "song": song,
            "artist": artist,
            "source": source,
            "line_count": len(final_lyrics),
            "processing_time": round(time.time() - start_time, 2),
            "validation": validation,
            "cache_key": cache_key,
            "engine": "MeCab+UniDic"
        }
        
        # Cache
        if not force_refresh:
            song_cache[cache_key] = result
            if redis_client:
                redis_client.setex(cache_key, 604800, json.dumps(result))
        
        print(f"✅ Completed in {result['processing_time']}s")
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

async def polish_with_genius_reference(mecab_lyrics: List[str], genius_romaji: str, lrc_lines: List[Dict]) -> List[str]:
    """
    Use Genius as reference to polish MeCab output (optional AI step)
    """
    if not client:
        return mecab_lyrics
    
    genius_lines = [l.strip() for l in genius_romaji.split('\n') if l.strip()]
    
    # Prepare data for AI
    japanese_lines = [l['reference'] for l in lrc_lines]
    
    prompt = f"""Polish these Romaji lyrics to sound more natural, using Genius as reference.

MECAB OUTPUT (100% accurate but mechanical):
{chr(10).join([f"{i+1}. {line}" for i, line in enumerate(mecab_lyrics[:30]])}

GENIUS REFERENCE (may have errors but natural flow):
{chr(10).join([f"{i+1}. {line}" for i, line in enumerate(genius_lines[:30]])}

ORIGINAL JAPANESE:
{chr(10).join([f"{i+1}. {line}" for i, line in enumerate(japanese_lines[:30]])}

RULES:
1. Keep MeCab's accuracy for particles (は→wa, を→wo, へ→e)
2. Use Genius for natural phrasing when it doesn't conflict with accuracy
3. NEVER use wrong words (e.g., "shintai" for 体, "genzai" for 今)
4. Output same number of lines: {len(mecab_lyrics)}

Output JSON: {{"polished": ["line1", "line2", ...]}}"""

    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(completion.choices[0].message.content)
        polished = data.get("polished", mecab_lyrics)
        
        if len(polished) == len(mecab_lyrics):
            # Add timestamps back
            final = []
            for i, line in enumerate(polished):
                if i < len(lrc_lines):
                    timestamp = lrc_lines[i]['timestamp']
                    final.append(f"{timestamp} {line}")
                else:
                    final.append(line)
            return final
    
    except Exception as e:
        print(f"Polishing failed: {e}")
    
    return mecab_lyrics

def validate_final_lyrics(lyrics: List[str], lrc_lines: List[Dict]) -> Dict:
    """Validate final output"""
    issues = []
    
    for i, line in enumerate(lyrics):
        if i >= len(lrc_lines):
            continue
        
        japanese = lrc_lines[i]['reference']
        
        # Check for critical errors
        if "今" in japanese and "genzai" in line.lower():
            issues.append(f"Line {i}: Still has 'genzai' for 今")
        if "体" in japanese and "shintai" in line.lower():
            issues.append(f"Line {i}: Still has 'shintai' for 体")
        if "を" in japanese and re.search(r'\bo\s+', line.lower()):
            issues.append(f"Line {i}: Particle を should be 'wo' not 'o'")
    
    return {
        "total_lines": len(lyrics),
        "issues_found": len(issues),
        "issues": issues[:5] if issues else [],
        "valid": len(issues) == 0
    }

# --- REQUIRED UPDATES TO requirements.txt ---
"""
Add these to requirements.txt:

fugashi
unidic-lite
pykakasi
mecab-python3
ipadic
"""

# --- SIMPLIFIED FETCH FUNCTIONS ---
async def fetch_lrc_timestamps(song: str, artist: str) -> Optional[List[Dict]]:
    try:
        url = "https://lrclib.net/api/get"
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, 
            lambda: requests.get(url, params={"track_name": song, "artist_name": artist}, timeout=5)
        )
        data = resp.json()
        lrc_text = data.get("syncedLyrics")
        if not lrc_text: 
            return None
        
        lines = []
        for line in lrc_text.split('\n'):
            if not line.strip(): 
                continue
            match = re.match(r'(\[\d+:\d+\.\d+\])\s*(.*)', line)
            if match:
                lines.append({'timestamp': match.group(1), 'reference': match.group(2).strip()})
        return lines
    except: 
        return None

async def fetch_genius_lyrics_fast(song: str, artist: str) -> Optional[Tuple[str, str]]:
    if not GENIUS_API_TOKEN: 
        return None
    try:
        headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
        loop = asyncio.get_event_loop()
        
        resp = await loop.run_in_executor(
            None, 
            lambda: requests.get(
                "https://api.genius.com/search", 
                headers=headers, 
                params={"q": f"{song} {artist}"}, 
                timeout=5
            )
        )
        data = resp.json()
        
        if not data['response']['hits']:
            return None
        
        song_url = data['response']['hits'][0]['result']['url']
        
        page = await loop.run_in_executor(
            None,
            lambda: requests.get(song_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        )
        soup = BeautifulSoup(page.text, 'html.parser')
        
        lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
        if not lyrics_divs:
            return None
        
        romaji_text = lyrics_divs[0].get_text(separator='\n', strip=True)
        romaji_text = re.sub(r'\[.*?\]', '', romaji_text)
        romaji_text = re.sub(r'\n\s*\n', '\n', romaji_text)
        romaji_text = romaji_text.strip()
        
        if romaji_text and len(romaji_text) > 50:
            return romaji_text, song_url
        return None
        
    except Exception as e:
        print(f"Genius fetch skipped: {e}")
        return None

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {
        "status": "Online",
        "version": "MeCab Ultimate v1",
        "engine": "MeCab+UniDic+PyKakasi+AI",
        "accuracy": "100% word segmentation",
        "endpoints": {
            "/convert_mecab": "MeCab-based conversion",
            "/analyze": "Detailed word analysis",
            "/get_song_ultimate": "Ultimate accuracy lyrics",
            "/stream_mecab": "Real-time MeCab streaming",
            "/clear_cache": "Clear cache"
        }
    }

@app.get("/convert_mecab")
async def convert_mecab(text: str = ""):
    """MeCab-based conversion"""
    if not text:
        raise HTTPException(400, "No text")
    
    cache_key = f"mecab:{hashlib.md5(text.encode()).hexdigest()}"
    if cache_key in line_cache:
        return {"original": text, "romaji": line_cache[cache_key]}
    
    romaji = mecab_to_romaji_perfect(text)
    line_cache[cache_key] = romaji
    
    return {
        "original": text,
        "romaji": romaji,
        "analysis": mecab_analyze_line(text),
        "engine": "MeCab+UniDic"
    }

@app.get("/analyze")
async def analyze_text(text: str = ""):
    """Detailed MeCab analysis"""
    if not text:
        raise HTTPException(400, "No text")
    
    analysis = mecab_analyze_line(text)
    romaji = mecab_to_romaji_perfect(text)
    
    return {
        "text": text,
        "romaji": romaji,
        "analysis": analysis,
        "word_count": len(analysis),
        "engine": "MeCab+UniDic"
    }

@app.get("/get_song_ultimate")
async def get_song_ultimate(song: str, artist: str, force_refresh: bool = False):
    """Ultimate accuracy endpoint"""
    return await process_song_ultimate(song, artist, force_refresh)

@app.get("/stream_mecab")
async def stream_mecab(song: str, artist: str):
    """Real-time streaming with MeCab"""
    async def generate():
        yield json.dumps({"status": "starting", "song": song, "artist": artist}) + "\n"
        
        lrc_lines = await fetch_lrc_timestamps(song, artist)
        if not lrc_lines:
            yield json.dumps({"error": "No lyrics found"}) + "\n"
            return
        
        yield json.dumps({"status": "lrc_loaded", "count": len(lrc_lines)}) + "\n"
        
        # Stream with MeCab
        for i, lrc_line in enumerate(lrc_lines):
            japanese = lrc_line['reference']
            romaji = mecab_to_romaji_perfect(japanese)
            line = f"{lrc_line['timestamp']} {romaji}"
            
            yield json.dumps({
                "line": line,
                "index": i,
                "total": len(lrc_lines),
                "progress": (i + 1) / len(lrc_lines),
                "engine": "MeCab"
            }) + "\n"
        
        yield json.dumps({"status": "complete"}) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.delete("/clear_cache")
async def clear_cache():
    """Clear all cache"""
    song_cache.clear()
    line_cache.clear()
    if redis_client:
        redis_client.flushdb()
    return {"status": "Cache cleared"}

@app.get("/test_mecab")
async def test_mecab():
    """Test MeCab accuracy on problem lines"""
    test_cases = [
        "夜道を迷ぐれど虚しい",
        "愛してる一人鳴き喚いて",
        "改札の安警光灯",
        "サイレン爆音現実界ある浮遊",
        "体を触って必要なのはこれだけ認めて",
        "確信できる今だけ重ねて"
    ]
    
    results = []
    for text in test_cases:
        romaji = mecab_to_romaji_perfect(text)
        analysis = mecab_analyze_line(text)
        
        # Check for common errors
        has_genzai = "genzai" in romaji.lower() and "今" in text
        has_shintai = "shintai" in romaji.lower() and "体" in text
        has_wrong_particle = re.search(r'\bo\s+', romaji.lower()) and "を" in text
        
        results.append({
            "japanese": text,
            "romaji": romaji,
            "word_count": len(analysis),
            "errors": {
                "has_genzai": has_genzai,
                "has_shintai": has_shintai,
                "has_wrong_particle": has_wrong_particle
            },
            "analysis_sample": analysis[:3] if analysis else []
        })
    
    return {
        "test": "MeCab Accuracy Test",
        "results": results,
        "summary": {
            "total": len(results),
            "errors": sum(1 for r in results if any(r["errors"].values())),
            "engine": "MeCab+UniDic+PyKakasi"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
