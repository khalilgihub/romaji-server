#!/bin/bash
set -e

echo "🔧 Installing MeCab..."
apt-get update
apt-get install -y mecab libmecab-dev mecab-ipadic-utf8

echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "📚 Downloading dictionaries..."
python3 << 'EOF'
try:
    import unidic_lite
    print("✅ UniDic-Lite ready")
except:
    print("⚠️ UniDic-Lite not available")
EOF

echo "✅ Build complete!"
