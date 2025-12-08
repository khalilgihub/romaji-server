from fastapi import FastAPI, HTTPException
import google.generativeai as genai
import requests
import os

app = FastAPI()

# --- 1. SETUP AI ---
API_KEY = os.environ.get("GEMINI_API_KEY")

# Variable to hold the working model
model = None

# --- SMART MODEL SELECTOR ---
def setup_ai():
    global model
    if not API_KEY:
        print("CRITICAL ERROR: GEMINI_API_KEY is missing!")
        return

    genai.configure(api_key=API_KEY)
    
    found_model_name = None

    # Ask Google what models are actually available for this Key
    try:
        print("Searching for available AI models...")
        for m in genai.list_models():
            # We only want models that can generate text
            if 'generateContent' in m.supported_generation_methods:
                # Prefer 'flash' if available (it's fast)
                if 'flash' in m.name:
                    found_model_name = m.name
                    break
                # Fallback to 'pro' if flash isn't there
                elif 'pro' in m.name and not found_model_name:
                    found_model_name = m.name
        
        # If the loop didn't find anything specific, default to gemini-pro
        if not found_model_name:
            found_model_name = 'models/gemini-pro'

        print(f"--- SUCCESS: Using Model '{found_model_name}' ---")
        model = genai.GenerativeModel(found_model_name)

    except Exception as e:
        print(f"Error finding model: {e}")
        # absolute fallback
        model = genai.GenerativeModel('gemini-pro')

# Run setup immediately
setup_ai()

# --- 2. MEMORY CACHE ---
line_cache = {}

@app.get("/convert")
async def convert_romaji(text: str = ""):
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    # Check Cache
    if text in line_cache:
        return {"original": text, "romaji": line_cache[text]}

    if not model:
        # If model failed to load, return original text (don't crash)
        return {"original": text, "romaji": text}

    try:
        prompt = (
            f"Convert this Japanese text to Romaji (Hepburn). "
            f"Return strictly ONLY the romaji. Text: {text}"
        )
        # Use async generation
        response = await model.generate_content_async(prompt)
        result = response.text.strip()
        
        # Save to Cache
        line_cache[text] = result
        
        return {"original": text, "romaji": result}

    except Exception as e:
        print(f"AI Conversion Error: {e}")
        return {"original": text, "romaji": text} # Fallback

@app.get("/prepare_song")
async def prepare_song(song: str, artist: str):
    if not model:
        raise HTTPException(status_code=500, detail="Server AI not configured.")

    url = "https://lrclib.net/api/get"
    try:
        resp = requests.get(url, params={"track_name": song, "artist_name": artist})
        data = resp.json()
        raw_lyrics = data.get("syncedLyrics") or data.get("plainLyrics")
    except:
        raise HTTPException(status_code=404, detail="Lyrics not found.")

    if not raw_lyrics:
        raise HTTPException(status_code=404, detail="Lyrics content is empty.")

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
