import os
import google.generativeai as genai
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
# Replace this with your actual API Key
API_KEY = "YOUR_API_KEY_HERE"
genai.configure(api_key=API_KEY)

# Function to find the best available model automatically
def get_best_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name:
                    return m.name
        return 'models/gemini-pro' # Fallback if flash isn't found
    except:
        return 'models/gemini-pro'

# Set the model once at startup
MODEL_NAME = get_best_model()
print(f"--- USING AI MODEL: {MODEL_NAME} ---")

model = genai.GenerativeModel(MODEL_NAME)

@app.route('/convert', methods=['GET'])
def convert_to_romaji():
    text = request.args.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # We tell the AI strictly what to do to avoid extra talking
        prompt = f"Convert this Japanese text to Romaji. Return ONLY the Romaji. Text: {text}"
        
        response = model.generate_content(prompt)
        
        # .text gives the result
        romaji_text = response.text.strip()
        
        return jsonify({
            "original": text,
            "romaji": romaji_text
        })

    except Exception as e:
        print(f"AI Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
