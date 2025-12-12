@app.get("/debug_conversion")
async def debug_conversion(text: str):
    """Debug a specific line to see detailed breakdown"""
    analysis = mecab_analyze_line_improved(text, tagger, kakasi_converter, DICTIONARY_TYPE)
    romaji = mecab_to_romaji_perfect_v2(text, tagger, kakasi_converter, DICTIONARY_TYPE)
    
    return {
        "input": text,
        "final_romaji": romaji,
        "word_by_word": [
            {
                "japanese": w.surface,
                "reading": w.reading,
                "romaji": w.romaji,
                "pos": w.pos
            } for w in analysis
        ],
        "dict_type": DICTIONARY_TYPE
    }
