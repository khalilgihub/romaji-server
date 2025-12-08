from fastapi import FastAPI
import cutlet
import re

app = FastAPI()

# Initialize with the heavy dictionary
katsu = cutlet.Cutlet('hepburn')
katsu.use_foreign_spelling = False 

# --- TYPE A: The "Impossible" Fixes (Artistic Readings) ---
# You MUST add these manually because they break language rules.
# Keep this list for the famous ones.
katsu.add_exception("現在", "ima")      
katsu.add_exception("未来", "mirai")    
katsu.add_exception("宇宙", "sora")     
katsu.add_exception("明日", "ashita")   
katsu.add_exception("永久", "towa")     

@app.get("/")
def home():
    return {"status": "Online"}

@app.get("/convert")
def convert_lyrics(text: str = ""):
    if not text:
        return {"original": "", "romaji": ""}
    
    # 1. Basic Conversion
    romaji = katsu.romaji(text)
    
    # 2. Apply "Smart Formatting" (Fixes spacing automatically)
    romaji = smart_format(romaji)

    return {"original": text, "romaji": romaji}

def smart_format(text: str) -> str:
    """
    Automatically fixes common spacing annoyances for ALL lyrics.
    """
    # 1. Combine "te/ta/de/da" forms (e.g., "kasane te" -> "kasanete")
    # Matches any word ending in a vowel + space + te/ta/de/da
    text = re.sub(r'([aeiou]) (te|ta|de|da)\b', r'\1\2', text)

    # 2. Combine negative forms (e.g., "waka ra nai" -> "wakaranai")
    text = re.sub(r' (nai|naka|zu)\b', r'\1', text)

    # 3. Combine continuous forms (e.g., "shi te iru" -> "shiteiru")
    text = text.replace("te iru", "teiru")
    text = text.replace("de iru", "deiru")

    # 4. Particles (Optional style preference)
    # Some people prefer "wo" over "o" for the particle
    text = re.sub(r'\bwo\b', 'o', text) 

    # 5. Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
        
    return text
