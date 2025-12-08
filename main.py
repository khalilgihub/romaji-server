from fastapi import FastAPI, HTTPException
import google.generativeai as genai
import requests
import os

app = FastAPI()

# --- 1. SETUP AI ---
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY is missing!")
    model = None
else:
    genai.configure(api_key=API_KEY)
    # Flash is the fastest model for both tasks
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("--- AI INITIALIZED (Gemini 1.5 Flash) ---")

# --- 2. MEMORY CACHE (For the old line-by-line method) ---
# Saves results so we don't ask AI twice for the same line
line_cache = {}

# =================================================================
# ROUTE 1: THE "OLD" WAY (Restored so your app works now)
# =================================================================
@app.get("/convert")
async def convert_romaji(text: str = ""):
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    # 1. Check Cache (Instant)
    if text in line_cache:
        return {"original": text, "romaji": line_cache[text]}

    if not model:
        raise HTTPException(status_code=500, detail="Server Error: API Key missing.")

    try:
        # 2. Ask AI (Optimized for speed)
        prompt = (
            f"Convert this Japanese text to Romaji (Hepburn). "
            f"Return strictly ONLY the romaji. Text: {text}"
        )
        response = await model.generate_content_async(prompt)
        result = response.text.strip()
        
        # 3. Save to Cache
        line_cache[text] = result
        
        return {"original": text, "romaji": result}

    except Exception as e:
        print(f"Error: {e}")
        return {"original": text, "romaji": text} # Fallback to original if error

# =================================================================
# ROUTE 2: THE "NEW" WAY (Batch Processing)
# Use this when you update your app to load the whole song at start
# =================================================================
@app.get("/prepare_song")
async def prepare_song(song: str, artist: str):
    if not model:
        raise HTTPException(status_code=500, detail="Server AI not configured.")

    # 1. Get raw lyrics from internet
    url = "https://lrclib.net/api/get"
    try:
        resp = requests.get(url, params={"track_name": song, "artist_name": artist})
        resp.raise_for_status()
        data = resp.json()
        raw_lyrics = data.get("syncedLyrics") or data.get("plainLyrics")
    except:
        raise HTTPException(status_code=404, detail="Lyrics not found.")

    if not raw_lyrics:
        raise HTTPException(status_code=404, detail="Lyrics content is empty.")

    # 2. Convert WHOLE song at once
    prompt = (
        f"Convert the Japanese text in these lyrics to Romaji (Hepburn). "
        f"KEEP the timestamps [00:00.00] exactly as they are. "
        f"Do NOT output the original Japanese. Only the Romaji lines.\n\n"
        f"Lyrics:\n{raw_lyrics}"
    )

    try:
        response = await model.generate_content_async(prompt)
        return {
            "status": "ready",
            "song": song,
            "lyrics": response.text.strip()
        }
    except Exception as e:
        return {"error": str(e), "lyrics": raw_lyrics}
