from fastapi import FastAPI, HTTPException
import google.generativeai as genai
import os

app = FastAPI()

# --- 1. SETUP AI ---
# PASTE YOUR API KEY HERE
API_KEY = "YOUR_API_KEY_HERE" 
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

MODEL_NAME = get_best_model()
model = genai.GenerativeModel(MODEL_NAME)

# --- 2. THE SERVER ROUTE ---
@app.get("/convert")
async def convert_romaji(text: str = ""):
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

# Note: No app.run() needed here. Uvicorn handles that automatically on Render.
