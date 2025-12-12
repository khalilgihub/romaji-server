#!/bin/bash
set -e

echo "🔧 Installing MeCab (SYSTEM PACKAGE)..."
apt-get update
apt-get install -y --no-install-recommends \
    mecab \
    mecab-ipadic-utf8 \
    libmecab-dev \
    && rm -rf /var/lib/apt/lists/*

echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Build complete!"
