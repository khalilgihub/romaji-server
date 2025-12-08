from fastapi import FastAPI
import cutlet

app = FastAPI()

# Load the converter once (Hepburn style)
katsu = cutlet.Cutlet('hepburn')
katsu.use_foreign_spelling = False 

@app.get("/")
def home():
    return {"status": "Online"}

@app.get("/convert")
def convert_lyrics(text: str = ""):
    if not text:
        return {"original": "", "romaji": ""}
    
    # The magic line that converts Kanji -> Romaji
    romaji = katsu.romaji(text)
    return {"original": text, "romaji": romaji}