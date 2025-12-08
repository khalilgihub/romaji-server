from fastapi import FastAPI
import google.generativeai as genai
import os

app = FastAPI()

# --- SETUP ---
# Your Google API Key
API_KEY = "AIzaSyBXPO9hlH1ueZ_UOOpHtElLVrMCa75zV9w"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

@app.get("/")
def home():
    return {"status": "AI Server Online"}

@app.get("/convert")
def convert_lyrics(text: str = ""):
    if not text:
        return {"original": "", "romaji": ""}
    
    try:
        # We ask Gemini to act like a professional lyricist
        prompt = f"""
        Convert this Japanese song lyric to Hepburn Romaji.
        Rules:
        1. Look for poetic readings (Gikun). Example: Read '現在' as 'ima' if it fits the context.
        2. Fix spacing issues. Combine verbs properly (e.g., 'kasanete', NOT 'kasane te').
        3. Output ONLY the romaji text. No explanations.
        
        Lyric: {text}
        """
        
        response = model.generate_content(prompt)
        romaji = response.text.strip()
        
        # Clean up any extra whitespace or newlines
        romaji = " ".join(romaji.split())

        return {"original": text, "romaji": romaji}

    except Exception as e:
        print(f"Error: {e}")
        # If AI fails (rare), return original text so app doesn't crash
        return {"original": text, "romaji": text}
