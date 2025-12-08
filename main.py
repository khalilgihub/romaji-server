from fastapi import FastAPI, HTTPException
import google.generativeai as genai
import requests
import os

app = FastAPI()

# --- SETUP AI ---
API_KEY = os.environ.get("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
    # Flash is perfect for handling large blocks of text quickly
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None
    print("CRITICAL: No API Key found.")

# --- HELPER: Search & Download Lyrics ---
def get_raw_lyrics_from_web(track_name: str, artist_name: str):
    url = "https://lrclib.net/api/get"
    params = {"track_name": track_name, "artist_name": artist_name}
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        # Prefer synced lyrics, fallback to plain
        return data.get("syncedLyrics") or data.get("plainLyrics")
    except Exception as e:
        print(f"Lyrics Search Error: {e}")
        return None

@app.get("/prepare_song")
async def prepare_song(song: str, artist: str):
    """
    1. Finds the lyrics.
    2. Converts the WHOLE song to Romaji at once.
    3. Returns the final ready-to-use data.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Server AI not configured.")

    # Step 1: Get the Japanese Lyrics
    raw_lyrics = get_raw_lyrics_from_web(song, artist)
    if not raw_lyrics:
        raise HTTPException(status_code=404, detail="Lyrics not found on LRCLIB.")

    # Step 2: Ask AI to convert the whole block
    # We ask it to KEEP the timestamps [00:12.34] but change the text
    prompt = (
        f"I will give you lyrics with timestamps. "
        f"Convert the Japanese text to Romaji (Hepburn). "
        f"KEEP the timestamps exactly as they are. "
        f"Do NOT output the original Japanese. Only the Romaji lines. "
        f"\n\nLyrics:\n{raw_lyrics}"
    )

    try:
        print(f"--- Processing entire song: {song} ---")
        response = await model.generate_content_async(prompt)
        converted_lyrics = response.text.strip()
        
        return {
            "status": "ready",
            "song": song,
            "artist": artist,
            "lyrics": converted_lyrics # The app just displays this directly!
        }

    except Exception as e:
        print(f"AI Batch Error: {e}")
        return {"error": str(e), "lyrics": raw_lyrics} # Fallback to raw if AI fails
