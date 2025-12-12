#!/bin/bash               # "Hey computer, run these as instructions"
set -e                    # "Stop if anything goes wrong"

echo "Installing MeCab..."  # "Print this message to screen"
apt-get update            # "Update the list of available software"
apt-get install -y mecab libmecab-dev mecab-ipadic-utf8  # "Install MeCab"

echo "Installing Python packages..."  # "Print this message"
pip install -r requirements.txt       # "Install Python packages from requirements.txt"

echo "Downloading dictionary..."      # "Print this message"
python -c "import unidic_lite; unidic_lite.download()"  # "Download Japanese dictionary"

echo "Done!"              # "Print 'Done!' when finished"
