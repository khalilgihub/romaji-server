from fastapi import FastAPI, HTTPException
import google.generativeai as genai
import os

app = FastAPI()

# --- 1. SETUP AI (SECURELY) ---
# This reads the key from Render's settings, not the file
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY is missing from Environment Variables!")
else:
    genai.configure(api_key=API_KEY)

# Helper to find the right model
def get_best_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name:
                    return m.name
        return 'models/gemini-pro'
    except:
        return 'models/gemini-pro'

# Set model (Handle case where API key is missing to prevent crash on startup)
if API_KEY:
    MODEL_NAME = get_best_model()
    model = genai.GenerativeModel(MODEL_NAME)
else:
    model = None

# --- 2. THE SERVER ROUTE ---
@app.get("/convert")
async def convert_romaji(text: str = ""):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server Error: API Key not configured.")
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        # Ask AI to convert
        prompt = f"Convert this Japanese text to Romaji. Return ONLY the Romaji. Text: {text}"
        response = model.generate_content(prompt)
        
        return {
            "original": text,
            "romaji": response.text.strip()
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
