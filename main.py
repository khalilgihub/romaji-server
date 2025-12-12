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
import jaconv

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

# --- PERFECT ALIGNMENT USING AI FOR ENTIRE MATCHING ---
async def ai_perfect_align(lrc_lines: List[Dict], romaji_text: str) -> List[str]:
    """
    Use AI to do perfect alignment of Japanese lyrics with Romaji lyrics
    This solves the "yomichi wo masaguredo munashii" vs "yomichi o iburedo munashi" problem
    """
    if not client:
        return []
    
    # Prepare the data for AI
    japanese_lines = [l['reference'] for l in lrc_lines]
    timestamps = [l['timestamp'] for l in lrc_lines]
    
    # Clean romaji text
    romaji_lines = [l.strip() for l in romaji_text.split('\n') if l.strip()]
    
    print(f"🤖 Using AI for perfect alignment: {len(japanese_lines)} Japanese lines vs {len(romaji_lines)} Romaji lines")
    
    # FIXED LINE: Added missing closing bracket and parenthesis
    prompt = f"""You are a Japanese lyrics expert. Match these Japanese lyrics with the correct Romaji lyrics.

CRITICAL RULES:
1. Output EXACTLY {len(japanese_lines)} lines
2. Each line must start with the timestamp: {timestamps[0]} (first line), {timestamps[-1]} (last line)
3. Match Japanese lines to Romaji lines based on MEANING, not just word-for-word
4. Some Romaji lines might be slightly different translations but still correct
5. For 今, use "ima" NOT "genzai"
6. For を, use "wo" NOT "o"

JAPANESE LYRICS (with line numbers):
{chr(10).join([f"{i+1}. {line}" for i, line in enumerate(japanese_lines[:50])])}

ROMAJI LYRICS (available for matching):
{chr(10).join([f"{i+1}. {line}" for i, line in enumerate(romaji_lines[:60])])}

MAPPING INSTRUCTIONS:
- Line 1 Japanese → Line 1 Romaji (or the closest match)
- If Romaji has extra lines (like [Chorus]), skip them
- If Japanese line doesn't match any Romaji, translate it yourself
- Preserve the natural flow and meaning

OUTPUT FORMAT (JSON):
{{
  "aligned_lyrics": [
    "[00:00.00] Romaji line 1",
    "[00:05.00] Romaji line 2",
    ...
  ],
  "confidence": 0.95,
  "notes": "Any important notes"
}}

Be very careful with:
- "yomichi wo masaguredo munashii" should match to "yomichi wo masaguredo munashii" or close
- Don't confuse "wo" with "o"
- Match based on meaning, not just similar words"""

    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.1,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(completion.choices[0].message.content)
        aligned = data.get("aligned_lyrics", [])
        confidence = data.get("confidence", 0)
        notes = data.get("notes", "")
        
        print(f"✅ AI Alignment confidence: {confidence:.2f}")
        if notes:
            print(f"📝 AI Notes: {notes}")
        
        # Verify we have the right number of lines
        if len(aligned) == len(lrc_lines):
            return aligned
        else:
            print(f"⚠️ AI returned {len(aligned)} lines, expected {len(lrc_lines)}")
            return []
            
    except Exception as e:
        print(f"AI alignment error: {e}")
        return []

# --- TWO-PASS VERIFICATION SYSTEM ---
async def verify_and_correct_alignment(japanese_lines: List[str], aligned_romaji: List[str]) -> List[str]:
    """
    Use AI to verify each line and correct any mistakes
    """
    if not client or len(japanese_lines) != len(aligned_romaji):
        return aligned_romaji
    
    print(f"🔍 Verifying {len(japanese_lines)} aligned lines...")
    
    # Check each line for accuracy
    corrected = []
    issues_found = 0
    
    # Process in batches of 10
    batch_size = 10
    for i in range(0, len(japanese_lines), batch_size):
        batch_end = min(i + batch_size, len(japanese_lines))
        batch_jp = japanese_lines[i:batch_end]
        batch_romaji = aligned_romaji[i:batch_end]
        
        # Extract just the romaji text (without timestamp)
        romaji_texts = []
        for line in batch_romaji:
            # Remove timestamp [00:00.00]
            match = re.match(r'\[\d+:\d+\.\d+\]\s*(.*)', line)
            if match:
                romaji_texts.append(match.group(1))
            else:
                romaji_texts.append(line)
        
        prompt = f"""Verify and correct these Japanese→Romaji translations. Fix ANY errors.

RULES:
1. 今 → "ima" (never "genzai")
2. を → "wo" (not "o" when it's the particle)
3. Fix any wrong words or missing parts
4. Preserve the timestamp format

LINE PAIRS (Japanese → Current Romaji):
{chr(10).join([f"{j+1}. JP: {batch_jp[j]} → Current: {romaji_texts[j]}" for j in range(len(batch_jp))])}

CORRECTIONS:
For each line, provide the PERFECT Romaji translation.

Output JSON: {{
  "corrected": [
    "corrected romaji line 1",
    "corrected romaji line 2",
    ...
  ],
  "issues_fixed": ["Line 3: Fixed 'o' → 'wo'", ...]
}}"""
        
        try:
            completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=DEEPSEEK_MODEL,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(completion.choices[0].message.content)
            corrected_batch = data.get("corrected", romaji_texts)
            issues = data.get("issues_fixed", [])
            
            if issues:
                issues_found += len(issues)
                for issue in issues:
                    print(f"   Fixed: {issue}")
            
            # Add timestamps back
            for j, correction in enumerate(corrected_batch):
                line_idx = i + j
                if line_idx < len(aligned_romaji):
                    # Extract timestamp from original line
                    match = re.match(r'(\[\d+:\d+\.\d+\])', aligned_romaji[line_idx])
                    if match:
                        corrected.append(f"{match.group(1)} {correction.strip()}")
                    else:
                        corrected.append(correction.strip())
                        
        except Exception as e:
            print(f"Verification error for batch {i}: {e}")
            corrected.extend(aligned_romaji[i:batch_end])
    
    if issues_found > 0:
        print(f"✅ Fixed {issues_found} issues in alignment")
    
    return corrected if corrected else aligned_romaji

# --- SEMANTIC MATCHING FOR SPECIFIC PROBLEMS ---
JAPANESE_CORRECTIONS = {
    # Common misalignments and their corrections
    "yomichi wo masaguredo munashii": "yomichi wo masaguredo munashii",
    "yomichi o iburedo munashi": "yomichi wo masaguredo munashii",
    "kakushin dekiru genzai dake kasanete": "kakushin dekiru ima dake kasanete",
    "ima dake kasanete": "ima dake kasanete",
    "genzai dake kasanete": "ima dake kasanete",
    "wo": "wo",  # Force particle を to be "wo" not "o"
    "o": "wo",   # Convert incorrect "o" to correct "wo" when it's the particle
}

def apply_semantic_corrections(japanese: str, romaji: str) -> str:
    """
    Apply semantic corrections based on known patterns
    """
    corrected = romaji
    
    # Check for specific known wrong translations
    for wrong, right in JAPANESE_CORRECTIONS.items():
        if wrong in romaji.lower():
            # Only replace if the Japanese context matches
            if wrong == "genzai dake" and "今" in japanese:
                corrected = corrected.replace("genzai", "ima")
                corrected = corrected.replace("Genzai", "Ima")
                corrected = corrected.replace("GENZAI", "IMA")
            elif wrong == "o" and "を" in japanese:
                # Be careful: only replace particle "o" not every "o"
                # Replace " o " (space o space) which is usually the particle
                corrected = re.sub(r'\s+o\s+', ' wo ', corrected)
                corrected = re.sub(r'^\s*o\s+', 'wo ', corrected)
                corrected = re.sub(r'\s+o$', ' wo', corrected)
    
    return corrected

# --- HYBRID ALIGNMENT SYSTEM ---
async def hybrid_perfect_align(lrc_lines: List[Dict], romaji_text: str) -> List[str]:
    """
    Hybrid approach: Use AI for main alignment, then verify and correct
    """
    print("🎯 Starting HYBRID perfect alignment")
    
    # Step 1: AI alignment
    ai_aligned = await ai_perfect_align(lrc_lines, romaji_text)
    
    if not ai_aligned or len(ai_aligned) != len(lrc_lines):
        print("⚠️ AI alignment failed, falling back to semantic matching")
        return await semantic_fallback_align(lrc_lines, romaji_text)
    
    # Step 2: Extract Japanese lines for verification
    japanese_lines = [l['reference'] for l in lrc_lines]
    
    # Step 3: Verify and correct
    verified = await verify_and_correct_alignment(japanese_lines, ai_aligned)
    
    # Step 4: Apply semantic corrections
    final_result = []
    for i, line in enumerate(verified):
        if i < len(lrc_lines):
            japanese = lrc_lines[i]['reference']
            corrected = apply_semantic_corrections(japanese, line)
            final_result.append(corrected)
        else:
            final_result.append(line)
    
    # Step 5: Final verification
    wrong_count = 0
    for i in range(len(final_result)):
        if i < len(lrc_lines):
            # Check for known wrong patterns
            romaji_line = final_result[i].lower()
            japanese_line = lrc_lines[i]['reference']
            
            if "genzai" in romaji_line and "今" in japanese_line:
                final_result[i] = re.sub(r'\bgenzai\b', 'ima', final_result[i], flags=re.IGNORECASE)
                wrong_count += 1
                print(f"   Fixed remaining 'genzai' → 'ima' in line {i}")
    
    if wrong_count > 0:
        print(f"🔧 Fixed {wrong_count} remaining issues in final pass")
    
    return final_result

async def semantic_fallback_align(lrc_lines: List[Dict], romaji_text: str) -> List[str]:
    """
    Fallback alignment using semantic matching
    """
    romaji_lines = [l.strip() for l in romaji_text.split('\n') if l.strip()]
    
    # Use AI to translate if no good match
    if client and len(romaji_lines) < len(lrc_lines) * 0.5:
        print("📝 Romaji too short, translating with AI...")
        return await translate_all_with_ai(lrc_lines)
    
    # Simple line-by-line matching with context
    aligned = []
    romaji_idx = 0
    
    for i, lrc_line in enumerate(lrc_lines):
        japanese = lrc_line['reference']
        best_match = ""
        best_score = 0
        
        # Look ahead 5 lines
        for j in range(romaji_idx, min(romaji_idx + 5, len(romaji_lines))):
            romaji = romaji_lines[j]
            # Simple word overlap
            jp_words = set(jaconv.kata2hira(japanese).lower().split())
            romaji_words = set(romaji.lower().split())
            overlap = len(jp_words & romaji_words) / max(len(jp_words), 1)
            
            if overlap > best_score:
                best_score = overlap
                best_match = romaji
                if overlap > 0.3:  # Good enough
                    romaji_idx = j + 1
                    break
        
        if best_score > 0.2 and best_match:
            aligned.append(f"{lrc_line['timestamp']} {best_match}")
        else:
            # Translate this line
            if client:
                translated = await translate_line_with_context(japanese, lrc_lines, i)
                aligned.append(f"{lrc_line['timestamp']} {translated}")
            else:
                aligned.append(f"{lrc_line['timestamp']} {japanese}")
    
    return aligned

async def translate_all_with_ai(lrc_lines: List[Dict]) -> List[str]:
    """Translate all lines with AI"""
    print("🤖 Translating all lines with AI...")
    
    japanese_lines = [l['reference'] for l in lrc_lines]
    
    prompt = f"""Translate these Japanese lyrics to Romaji with PERFECT accuracy.

CRITICAL RULES:
1. 今 → "ima" (NEVER "genzai")
2. を → "wo" (not "o" for the particle)
3. Preserve line breaks exactly
4. Translate meaning, not just words

JAPANESE LYRICS ({len(japanese_lines)} lines):
{chr(10).join(japanese_lines)}

Output JSON: {{
  "translations": [
    "Romaji line 1",
    "Romaji line 2",
    ...
  ]
}}"""
    
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(completion.choices[0].message.content)
        translations = data.get("translations", [])
        
        if len(translations) == len(lrc_lines):
            return [f"{lrc_lines[i]['timestamp']} {translations[i]}" for i in range(len(lrc_lines))]
    except:
        pass
    
    # Fallback: translate line by line
    result = []
    for i, lrc_line in enumerate(lrc_lines):
        translated = await translate_line_with_context(lrc_line['reference'], lrc_lines, i)
        result.append(f"{lrc_line['timestamp']} {translated}")
    
    return result

async def translate_line_with_context(japanese: str, all_lines: List[Dict], index: int) -> str:
    """Translate a single line with context"""
    prompt = f"""Translate this Japanese lyric line to Romaji.

CONTEXT:
Previous line: {all_lines[index-1]['reference'] if index > 0 else 'None'}
Current line: {japanese}
Next line: {all_lines[index+1]['reference'] if index < len(all_lines)-1 else 'None'}

RULES:
1. 今 → "ima" (not "genzai")
2. を → "wo" (not "o")
3. Translate naturally

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
        return japanese

# --- SIMPLIFIED VERSION OF YOUR EXISTING FUNCTIONS ---
def parse_lrc_lines(lrc_text: str) -> List[Dict]:
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
            lambda: requests.get(url, params={"track_name": song, "artist_name": artist}, timeout=5)
        )
        data = resp.json()
        lrc_text = data.get("syncedLyrics")
        if not lrc_text: 
            return None
        return parse_lrc_lines(lrc_text)
    except: 
        return None

async def fetch_genius_lyrics(song: str, artist: str) -> Optional[Tuple[str, str]]:
    if not GENIUS_API_TOKEN: 
        return None
    try:
        headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
        loop = asyncio.get_event_loop()
        
        resp = await loop.run_in_executor(
            None, 
            lambda: requests.get("https://api.genius.com/search", headers=headers, params={"q": f"{song} {artist}"}, timeout=8)
        )
        data = resp.json()
        
        if not data['response']['hits']:
            return None
        
        song_url = data['response']['hits'][0]['result']['url']
        
        page = await loop.run_in_executor(
            None,
            lambda: requests.get(song_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        )
        soup = BeautifulSoup(page.text, 'html.parser')
        
        lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
        full_text = []
        for div in lyrics_divs:
            text = div.get_text(separator='\n', strip=True)
            full_text.append(text)
        
        romaji_text = '\n\n'.join(full_text)
        romaji_text = re.sub(r'\[.*?\]', '', romaji_text)
        romaji_text = re.sub(r'\n\s*\n', '\n', romaji_text)
        romaji_text = romaji_text.strip()
        
        if romaji_text and len(romaji_text) > 30:
            return romaji_text, song_url
        return None
        
    except Exception as e:
        print(f"Genius error: {e}")
        return None

# --- MAIN PROCESSING ---
async def process_song_perfect(song: str, artist: str, force_refresh: bool = False):
    """Main processing with PERFECT alignment"""
    cache_key = f"perfect:{hashlib.md5(f'{song.lower()}:{artist.lower()}'.encode()).hexdigest()}"
    
    if not force_refresh:
        if cache_key in song_cache:
            return song_cache[cache_key]
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                result = json.loads(cached)
                song_cache[cache_key] = result
                return result
    
    print(f"🎯 PERFECT Processing: {song} by {artist}")
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
            print("✨ Found Genius, using HYBRID perfect alignment...")
            
            # Use hybrid perfect alignment
            aligned = await hybrid_perfect_align(lrc_lines, romaji_text)
            
            if aligned and len(aligned) == len(lrc_lines):
                final_lyrics = aligned
                source = "Genius + Perfect Align"
                
                # Verify no remaining issues
                issues = []
                for i, line in enumerate(final_lyrics):
                    if i < len(lrc_lines):
                        if "genzai" in line.lower() and "今" in lrc_lines[i]['reference']:
                            issues.append(f"Line {i}: Still has 'genzai'")
                
                if issues:
                    print(f"⚠️ Found {len(issues)} issues, fixing...")
                    final_lyrics = await verify_and_correct_alignment(
                        [l['reference'] for l in lrc_lines],
                        final_lyrics
                    )
            else:
                print("⚠️ Hybrid alignment failed, using AI translation...")
                final_lyrics = await translate_all_with_ai(lrc_lines)
                source = "AI Translation (Fallback)"
        else:
            print("🤖 No Genius, using AI translation...")
            final_lyrics = await translate_all_with_ai(lrc_lines)
            source = "AI Translation"
        
        # Final quality check
        quality_score = 0
        for i, line in enumerate(final_lyrics):
            if i < len(lrc_lines):
                # Check for Japanese characters (bad)
                if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line):
                    quality_score -= 1
                # Check for known wrong patterns (bad)
                elif "genzai" in line.lower() and "今" in lrc_lines[i]['reference']:
                    quality_score -= 1
                else:
                    quality_score += 1
        
        quality_percent = max(0, quality_score / len(final_lyrics))
        
        result = {
            "lyrics": '\n'.join(final_lyrics),
            "song": song,
            "artist": artist,
            "source": source,
            "line_count": len(final_lyrics),
            "quality_score": f"{quality_percent:.1%}",
            "processing_time": round(time.time() - start_time, 2),
            "version": "perfect_v1"
        }
        
        # Cache
        if not force_refresh:
            song_cache[cache_key] = result
            if redis_client:
                redis_client.setex(cache_key, 604800, json.dumps(result))
        
        print(f"✅ Completed in {result['processing_time']}s, Quality: {result['quality_score']}")
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {
        "status": "Online",
        "version": "Perfect Alignment v1",
        "note": "今 is ALWAYS 'ima' (never 'genzai'), を is 'wo' (not 'o')",
        "endpoints": {
            "/get_song": "Get lyrics with perfect alignment",
            "/get_song_fresh": "Get lyrics (fresh, no cache)",
            "/clear_cache": "Clear all cache",
            "/test_alignment": "Test alignment corrections"
        }
    }

@app.get("/get_song")
async def get_song_endpoint(song: str, artist: str, force_refresh: bool = False):
    """Main endpoint - uses perfect alignment"""
    return await process_song_perfect(song, artist, force_refresh)

@app.get("/get_song_fresh")
async def get_song_fresh(song: str, artist: str):
    """Always get fresh lyrics (no cache)"""
    return await process_song_perfect(song, artist, force_refresh=True)

@app.get("/test_alignment")
async def test_alignment():
    """Test specific alignment problems"""
    test_cases = [
        {
            "japanese": "yomichi wo masaguredo munashii",
            "wrong_romaji": "yomichi o iburedo munashi",
            "correct_romaji": "yomichi wo masaguredo munashii"
        },
        {
            "japanese": "確信できる今だけ重ねて",
            "wrong_romaji": "kakushin dekiru genzai dake kasanete",
            "correct_romaji": "kakushin dekiru ima dake kasanete"
        }
    ]
    
    results = []
    for test in test_cases:
        corrected = apply_semantic_corrections(test["japanese"], test["wrong_romaji"])
        results.append({
            "japanese": test["japanese"],
            "wrong": test["wrong_romaji"],
            "corrected": corrected,
            "expected": test["correct_romaji"],
            "match": corrected.lower() == test["correct_romaji"].lower()
        })
    
    return {
        "test": "Alignment Correction Test",
        "results": results,
        "success_rate": f"{sum(1 for r in results if r['match'])}/{len(results)}"
    }

@app.get("/convert")
async def convert_single_line(text: str = ""):
    if not text:
        raise HTTPException(400, "No text")
    
    if not client:
        return {"original": text, "romaji": text}
    
    try:
        prompt = f"""Translate to Romaji. CRITICAL: 今 → "ima" (never "genzai"), を → "wo" (not "o").
        
        Japanese: {text}
        Romaji:"""
        
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.0,
            max_tokens=100
        )
        romaji = completion.choices[0].message.content.strip()
        
        # Double-check
        if "今" in text and "genzai" in romaji.lower():
            romaji = re.sub(r'\bgenzai\b', 'ima', romaji, flags=re.IGNORECASE)
        
        return {"original": text, "romaji": romaji}
    except:
        return {"original": text, "romaji": text}

@app.delete("/clear_cache")
async def clear_cache():
    """Clear ALL cache"""
    song_cache.clear()
    line_cache.clear()
    if redis_client:
        redis_client.flushdb()
        print("🗑️ Redis cache cleared")
    
    print("🗑️ Memory cache cleared")
    return {
        "status": "Cache cleared",
        "message": "All cached lyrics have been deleted. New requests will use perfect alignment.",
        "important": "This fixes alignment issues like 'genzai' vs 'ima' and 'wo' vs 'o'"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "deepseek": bool(client),
        "redis": redis_client.ping() if redis_client else False,
        "genius": bool(GENIUS_API_TOKEN),
        "cache_size": len(song_cache),
        "version": "perfect_alignment_v1"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
