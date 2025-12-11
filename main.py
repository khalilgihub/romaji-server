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

# --- CRITICAL FIX: WORD PREFERENCE MAPPING ---
JAPANESE_TO_ROMAJI_PREFERENCES = {
    # Force "ima" for 今 in ALL cases for songs
    '今': ['ima', 'now', 'present'],
    '現在': ['genzai', 'current', 'present time'],
    '私': ['watashi', 'I', 'me'],
    '僕': ['boku', 'I', 'me (male)'],
    '俺': ['ore', 'I', 'me (male casual)'],
    'あなた': ['anata', 'you'],
    '君': ['kimi', 'you'],
    '愛': ['ai', 'love'],
    '恋': ['koi', 'love', 'romance'],
    '心': ['kokoro', 'heart', 'mind'],
    '言葉': ['kotoba', 'word', 'language'],
    '世界': ['sekai', 'world'],
    '夢': ['yume', 'dream'],
    '未来': ['mirai', 'future'],
    '過去': ['kako', 'past'],
    '時間': ['jikan', 'time'],
    '場所': ['basho', 'place'],
    '声': ['koe', 'voice'],
    '手': ['te', 'hand'],
    '目': ['me', 'eye'],
    '涙': ['namida', 'tear'],
    '笑顔': ['egao', 'smile'],
    '幸せ': ['shiawase', 'happiness'],
}

def normalize_japanese(text: str) -> str:
    """Normalize Japanese text"""
    if not text:
        return ""
    try:
        text = jaconv.normalize(text)
        text = jaconv.kata2hira(text)
        text = unicodedata.normalize('NFKC', text.lower())
        text = re.sub(r'[「」【】『』()\[\]{}、。！？・]', '', text)
        return text.strip()
    except:
        text = unicodedata.normalize('NFKC', text.lower())
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

def get_preferred_romaji_for_word(japanese_word: str) -> str:
    """Get the preferred Romaji translation for a Japanese word"""
    return JAPANESE_TO_ROMAJI_PREFERENCES.get(japanese_word, [japanese_word])[0]

def japanese_to_preferred_romaji(japanese: str) -> str:
    """Convert Japanese to preferred Romaji for matching"""
    result = japanese
    for jp_word in JAPANESE_TO_ROMAJI_PREFERENCES:
        if jp_word in result:
            preferred = get_preferred_romaji_for_word(jp_word)
            result = result.replace(jp_word, preferred)
    return result.lower()

def calculate_similarity_with_preferences(japanese: str, romaji: str) -> Tuple[float, str, str]:
    """
    Calculate similarity with STRICT word preferences
    Returns: (score, analysis, suggested_correction)
    """
    # Step 1: Check if romaji contains unwanted translations
    romaji_lower = romaji.lower()
    
    # REJECT romaji lines that contain unwanted translations
    unwanted_patterns = [
        (r'\bgenzai\b', '今 should be "ima" not "genzai"'),
        (r'\bpresent\b', '今 should be "ima" not "present"'),
        (r'\bcurrent\b', '現在 should be "genzai" not "current"'),
    ]
    
    for pattern, reason in unwanted_patterns:
        if re.search(pattern, romaji_lower):
            return 0.0, f"rejected:{reason}", ""
    
    # Step 2: Convert Japanese to preferred Romaji
    preferred_romaji = japanese_to_preferred_romaji(japanese)
    norm_jp = normalize_japanese(japanese)
    norm_romaji = romaji_lower.strip()
    norm_preferred = preferred_romaji.lower()
    
    # Step 3: Check direct matches first
    if norm_preferred in norm_romaji or norm_romaji in norm_preferred:
        return 1.0, "direct_preferred_match", romaji
    
    # Step 4: Check for preferred words in romaji
    score = 0
    analysis_parts = []
    
    # Word overlap with preferred romaji
    pref_words = set(norm_preferred.split())
    romaji_words = set(norm_romaji.split())
    word_overlap = len(pref_words & romaji_words)
    
    if word_overlap > 0:
        score += 0.3
        analysis_parts.append(f"words:{word_overlap}")
    
    # Sequence matching
    seq_score = SequenceMatcher(None, norm_preferred, norm_romaji).ratio()
    if seq_score > 0.3:
        score += seq_score * 0.7
        analysis_parts.append(f"seq:{seq_score:.2f}")
    
    # Check for key preferred words
    for jp_word, prefs in JAPANESE_TO_ROMAJI_PREFERENCES.items():
        if jp_word in japanese:
            # Check if ANY of the preferred translations are in romaji
            found = False
            for pref in prefs:
                if pref.lower() in romaji_lower:
                    found = True
                    score += 0.2
                    analysis_parts.append(f"has_pref:{pref}")
                    break
    
    analysis = "|".join(analysis_parts) if analysis_parts else "no_match"
    
    # Step 5: Suggest correction if needed
    suggested_correction = ""
    if score < 0.6:
        # Generate a corrected version
        words = romaji_lower.split()
        corrected_words = []
        for word in words:
            corrected = word
            # Fix common mistakes
            if word == "genzai" and "今" in japanese:
                corrected = "ima"
            elif word == "present" and "今" in japanese:
                corrected = "ima"
            elif word == "current" and "現在" in japanese:
                corrected = "genzai"
            corrected_words.append(corrected)
        suggested_correction = " ".join(corrected_words)
    
    return min(score, 1.0), analysis, suggested_correction

# --- FETCH LRC TIMESTAMPS ---
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

# --- FETCH GENIUS LYRICS ---
async def fetch_genius_lyrics(song: str, artist: str) -> Optional[Tuple[str, str]]:
    if not GENIUS_API_TOKEN: 
        return None
    
    try:
        headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
        loop = asyncio.get_event_loop()
        
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
        
        song_url = data['response']['hits'][0]['result']['url']
        
        page = await loop.run_in_executor(
            None,
            lambda: requests.get(
                song_url, 
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=8
            )
        )
        soup = BeautifulSoup(page.text, 'html.parser')
        
        # Find lyrics
        lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
        if not lyrics_divs:
            return None
        
        romaji_text = ""
        for div in lyrics_divs:
            text = div.get_text(separator='\n', strip=True)
            # Check if it looks like Romaji
            latin = len(re.findall(r'[a-zA-Z]', text))
            japanese = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))
            if latin > japanese:
                romaji_text += text + "\n\n"
        
        romaji_text = re.sub(r'\[.*?\]', '', romaji_text)
        romaji_text = re.sub(r'\n\s*\n', '\n', romaji_text)
        romaji_text = romaji_text.strip()
        
        if romaji_text and len(romaji_text) > 30:
            return romaji_text, song_url
        return None
        
    except Exception as e:
        print(f"Genius error: {e}")
        return None

# --- STRICT ALIGNMENT WITH WORD PREFERENCE ENFORCEMENT ---
async def strict_align_with_preferences(lrc_lines: List[Dict], romaji_text: str) -> List[str]:
    """Strict alignment that ENFORCES word preferences"""
    
    romaji_lines = [l.strip() for l in romaji_text.split('\n') if l.strip()]
    if not romaji_lines:
        return []
    
    print(f"🔍 Strict alignment: {len(lrc_lines)} JP vs {len(romaji_lines)} Romaji")
    print("⚠️ ENFORCING: 今 → 'ima' (never 'genzai')")
    
    aligned = []
    romaji_idx = 0
    corrections_made = 0
    
    for lrc_idx, lrc_line in enumerate(lrc_lines):
        japanese = lrc_line['reference']
        
        # Skip if no Japanese text
        if not japanese or not japanese.strip():
            aligned.append(f"{lrc_line['timestamp']} ")
            continue
        
        best_score = 0
        best_line = ""
        best_idx = -1
        best_analysis = ""
        needs_correction = False
        
        # Search nearby romaji lines
        search_start = max(0, romaji_idx - 2)
        search_end = min(len(romaji_lines), romaji_idx + 5)
        
        for i in range(search_start, search_end):
            romaji_line = romaji_lines[i]
            score, analysis, suggested = calculate_similarity_with_preferences(japanese, romaji_line)
            
            # DEBUG: Log what we're finding
            if "今" in japanese and ("genzai" in romaji_line.lower() or "ima" in romaji_line.lower()):
                print(f"   Line {lrc_idx}: JP='{japanese}' → Romaji='{romaji_line}' (score={score:.2f}, {analysis})")
            
            if score > best_score:
                best_score = score
                best_line = romaji_line
                best_idx = i
                best_analysis = analysis
                needs_correction = bool(suggested)
        
        # Apply alignment
        if best_score > 0.4:  # Even low scores if they're the best we have
            final_line = best_line
            
            # AUTO-CORRECT if needed
            if needs_correction and "今" in japanese:
                # Force replace "genzai" with "ima"
                if "genzai" in final_line.lower():
                    final_line = re.sub(r'\bgenzai\b', 'ima', final_line, flags=re.IGNORECASE)
                    corrections_made += 1
                    print(f"   ✓ Auto-corrected 'genzai' → 'ima' in line {lrc_idx}")
                elif "present" in final_line.lower():
                    final_line = re.sub(r'\bpresent\b', 'ima', final_line, flags=re.IGNORECASE)
                    corrections_made += 1
                    print(f"   ✓ Auto-corrected 'present' → 'ima' in line {lrc_idx}")
            
            aligned.append(f"{lrc_line['timestamp']} {final_line}")
            romaji_idx = best_idx + 1
        else:
            # No match, use AI translation
            aligned.append(f"{lrc_line['timestamp']} {japanese}")  # Placeholder
    
    print(f"✅ Made {corrections_made} auto-corrections for word preferences")
    
    # Use AI to fix remaining Japanese lines
    if client:
        japanese_count = sum(1 for line in aligned if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line))
        if japanese_count > 0:
            print(f"🤖 Using AI to translate {japanese_count} remaining Japanese lines...")
            aligned = await ai_translate_remaining(aligned, lrc_lines, romaji_lines)
    
    return aligned

async def ai_translate_remaining(aligned: List[str], lrc_lines: List[Dict], romaji_lines: List[str]) -> List[str]:
    """AI translation for lines that didn't match"""
    
    # Find lines that need translation
    needs_translation = []
    for i, line in enumerate(aligned):
        if i < len(lrc_lines) and re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line):
            needs_translation.append({
                "index": i,
                "japanese": lrc_lines[i]['reference'],
                "timestamp": lrc_lines[i]['timestamp']
            })
    
    if not needs_translation:
        return aligned
    
    # Prepare batch for AI
    japanese_lines = [item["japanese"] for item in needs_translation]
    
    prompt = f"""TRANSLATE these Japanese lyrics to Romaji with STRICT RULES:

MANDATORY WORD CHOICES:
- 今 → ALWAYS "ima" (NEVER "genzai" or "present")
- 現在 → "genzai" (only for "current time")
- 私 → "watashi"
- あなた → "anata"
- 君 → "kimi"

IMPORTANT: If you see "今" in Japanese, output MUST contain "ima" not "genzai"!

Japanese lines ({len(japanese_lines)}):
{json.dumps(japanese_lines, ensure_ascii=False)}

Output JSON: {{"translations": ["romaji1", "romaji2", ...]}}
Be strict about word choices!"""
    
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        data = json.loads(completion.choices[0].message.content)
        translations = data.get("translations", [])
        
        # Apply translations
        result = aligned.copy()
        for idx, trans_item in enumerate(needs_translation):
            if idx < len(translations):
                line_idx = trans_item["index"]
                romaji = translations[idx].strip()
                
                # Double-check: Ensure "ima" not "genzai" for 今
                if "今" in trans_item["japanese"] and "genzai" in romaji.lower():
                    romaji = re.sub(r'\bgenzai\b', 'ima', romaji, flags=re.IGNORECASE)
                
                result[line_idx] = f"{trans_item['timestamp']} {romaji}"
        
        print(f"✅ AI translated {len(needs_translation)} lines")
        return result
        
    except Exception as e:
        print(f"AI translation failed: {e}")
        return aligned

# --- BATCH TRANSLATION WITH STRICT RULES ---
async def batch_translate_strict(japanese_lines: List[str]) -> List[str]:
    """Translation with zero tolerance for wrong word choices"""
    if not client or not japanese_lines:
        return japanese_lines
    
    prompt = f"""CRITICAL TRANSLATION TASK - MUST FOLLOW RULES:

NON-NEGOTIABLE RULES:
1. 今 → ALWAYS "ima" (NEVER "genzai", "present", or "current")
2. 現在 → "genzai" (only when explicitly "current time")
3. 私 → "watashi" (default)
4. あなた → "anata" 
5. 君 → "kimi"

These rules are ABSOLUTE. Never use "genzai" for 今 in song lyrics.

Japanese lines ({len(japanese_lines)}):
{json.dumps(japanese_lines, ensure_ascii=False)}

Output JSON: {{"translations": ["romaji1", "romaji2", ...]}}
Check each line for rule compliance!"""
    
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEEPSEEK_MODEL,
            temperature=0.0,  # Zero temperature for strict compliance
            response_format={"type": "json_object"}
        )
        data = json.loads(completion.choices[0].message.content)
        translations = data.get("translations", [])
        
        # Verify and fix if needed
        for i, (japanese, romaji) in enumerate(zip(japanese_lines, translations)):
            if "今" in japanese and "genzai" in romaji.lower():
                translations[i] = re.sub(r'\bgenzai\b', 'ima', romaji, flags=re.IGNORECASE)
                print(f"⚠️ Fixed AI mistake: 'genzai' → 'ima' in line {i}")
        
        return translations
        
    except Exception as e:
        print(f"Strict translation failed: {e}")
        return japanese_lines

# --- MAIN PROCESSING WITH FORCE REFRESH ---
async def process_song(song: str, artist: str, force_refresh: bool = False):
    """Main processing with cache bypass option"""
    cache_key = f"song_v2:{hashlib.md5(f'{song.lower()}:{artist.lower()}'.encode()).hexdigest()}"
    
    # Skip cache if force_refresh
    if not force_refresh:
        if cache_key in song_cache:
            print(f"📦 Using cached result (force_refresh={force_refresh})")
            return song_cache[cache_key]
        
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                result = json.loads(cached)
                song_cache[cache_key] = result
                print(f"📦 Using Redis cache (force_refresh={force_refresh})")
                return result
    
    print(f"🔄 Processing FRESH: {song} by {artist}")
    print("⚠️ WORD PREFERENCE ENFORCEMENT ACTIVE: 今 → 'ima' (not 'genzai')")
    start_time = time.time()
    
    try:
        # Get LRC
        lrc_lines = await fetch_lrc_timestamps(song, artist)
        if not lrc_lines:
            raise HTTPException(404, "No lyrics found")
        
        # Try Genius
        genius_result = await fetch_genius_lyrics(song, artist)
        
        final_lyrics = []
        source = ""
        
        if genius_result:
            romaji_text, _ = genius_result
            print("✨ Found Genius, aligning with STRICT preferences...")
            
            aligned = await strict_align_with_preferences(lrc_lines, romaji_text)
            
            # Check result
            bad_lines = []
            for i, line in enumerate(aligned):
                if "今" in lrc_lines[i]['reference'] and "genzai" in line.lower():
                    bad_lines.append(i)
            
            if bad_lines:
                print(f"⚠️ Still found {len(bad_lines)} lines with 'genzai' for 今, using AI translation...")
                translated = await batch_translate_strict([l['reference'] for l in lrc_lines])
                final_lyrics = [
                    f"{lrc_lines[i]['timestamp']} {translated[i]}"
                    for i in range(len(lrc_lines))
                ]
                source = "AI Translation (Strict)"
            else:
                final_lyrics = aligned
                source = "Genius + Strict Align"
        else:
            print("🤖 No Genius, using strict AI translation...")
            translated = await batch_translate_strict([l['reference'] for l in lrc_lines])
            final_lyrics = [
                f"{lrc_lines[i]['timestamp']} {translated[i]}"
                for i in range(len(lrc_lines))
            ]
            source = "AI Translation (Strict)"
        
        # Final verification
        for i, line in enumerate(final_lyrics):
            if i < len(lrc_lines) and "今" in lrc_lines[i]['reference']:
                if "genzai" in line.lower():
                    print(f"❌ CRITICAL ERROR: Line {i} still has 'genzai'!")
                    # Force fix
                    final_lyrics[i] = re.sub(r'\bgenzai\b', 'ima', line, flags=re.IGNORECASE)
        
        result = {
            "lyrics": '\n'.join(final_lyrics),
            "song": song,
            "artist": artist,
            "source": source,
            "line_count": len(final_lyrics),
            "processing_time": round(time.time() - start_time, 2),
            "cached": not force_refresh,
            "version": "v2_strict"
        }
        
        # Cache only if not forced refresh
        if not force_refresh:
            song_cache[cache_key] = result
            if redis_client:
                redis_client.setex(cache_key, 604800, json.dumps(result))
        
        print(f"✅ Completed in {result['processing_time']}s")
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

# --- ENDPOINTS WITH FORCE REFRESH ---
@app.get("/")
async def root():
    return {
        "status": "Online",
        "version": "2.0 - Strict Word Preferences",
        "note": "今 is ALWAYS 'ima' (never 'genzai')",
        "endpoints": {
            "/get_song": "Get lyrics (cached)",
            "/get_song_fresh": "Get lyrics (fresh, no cache)",
            "/clear_cache": "Clear all cache",
            "/debug_word": "Debug word matching"
        }
    }

@app.get("/get_song")
async def get_song_endpoint(song: str, artist: str, force_refresh: bool = False):
    """Main endpoint with optional force refresh"""
    return await process_song(song, artist, force_refresh)

@app.get("/get_song_fresh")
async def get_song_fresh(song: str, artist: str):
    """Always get fresh lyrics (no cache)"""
    return await process_song(song, artist, force_refresh=True)

@app.get("/debug_word")
async def debug_word(japanese: str):
    """Debug word preferences"""
    preferred = get_preferred_romaji_for_word(japanese)
    approx = japanese_to_preferred_romaji(japanese)
    
    return {
        "japanese": japanese,
        "preferred_romaji": preferred,
        "full_conversion": approx,
        "rule": JAPANESE_TO_ROMAJI_PREFERENCES.get(japanese, "No specific rule")
    }

@app.get("/convert")
async def convert_single_line(text: str = ""):
    if not text:
        raise HTTPException(400, "No text")
    
    # Special handling for 今
    if "今" in text:
        print(f"⚠️ Converting line with 今: {text}")
    
    if not client:
        return {"original": text, "romaji": text}
    
    try:
        prompt = f"""Translate to Romaji. CRITICAL: 今 → "ima" (never "genzai").
        
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
    """Clear ALL cache - DO THIS NOW to fix the issue"""
    song_cache.clear()
    line_cache.clear()
    if redis_client:
        redis_client.flushdb()
        print("🗑️ Redis cache cleared")
    
    print("🗑️ Memory cache cleared")
    return {
        "status": "Cache cleared",
        "message": "All cached lyrics have been deleted. New requests will use the strict word preferences.",
        "important": "This fixes the 'genzai' vs 'ima' issue"
    }

@app.get("/test_ima")
async def test_ima():
    """Test endpoint to verify 今 → ima"""
    test_cases = [
        "今だけ",
        "今、この瞬間",
        "今は未来",
        "今という時間",
        "今を生きる"
    ]
    
    results = []
    for jp in test_cases:
        if client:
            prompt = f"Translate to Romaji: 今 → 'ima' (not 'genzai'): {jp}"
            completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=DEEPSEEK_MODEL,
                temperature=0.0
            )
            romaji = completion.choices[0].message.content.strip()
        else:
            romaji = jp
        
        results.append({
            "japanese": jp,
            "romaji": romaji,
            "has_genzai": "genzai" in romaji.lower(),
            "has_ima": "ima" in romaji.lower()
        })
    
    return {
        "test": "今 → ima verification",
        "results": results,
        "summary": f"{sum(1 for r in results if not r['has_genzai'])}/{len(results)} correct"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
