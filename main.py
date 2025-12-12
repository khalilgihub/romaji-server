"""
FIXED MECAB ROMAJI CONVERSION FUNCTIONS
Replace the corresponding functions in your main code with these improved versions
"""

import fugashi
import pykakasi
import jaconv
import re
import logging
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WordAnalysis:
    surface: str
    reading: Optional[str]
    romaji: Optional[str]
    pos: Optional[str]
    pos_detail: Optional[str]
    base_form: Optional[str]
    dict_type: Optional[str]  # NEW: Track which dictionary is being used

# Global to track dictionary type
DICTIONARY_TYPE = None

def initialize_mecab_improved() -> tuple[fugashi.Tagger, str]:
    """Initialize MeCab and detect dictionary type"""
    global DICTIONARY_TYPE
    
    try:
        # Try UniDic first
        try:
            tagger = fugashi.Tagger('-r /dev/null -d /usr/lib/x86_64-linux-gnu/mecab/dic/unidic')
            DICTIONARY_TYPE = "unidic"
            logger.info("✅ MeCab + UniDic Loaded")
            return tagger, "unidic"
        except Exception:
            # Fallback to IPADIC
            tagger = fugashi.Tagger()
            DICTIONARY_TYPE = "ipadic"
            logger.info("✅ MeCab + IPADIC Loaded")
            return tagger, "ipadic"
    except Exception as e:
        logger.error(f"❌ MeCab failed: {e}")
        DICTIONARY_TYPE = None
        return None, None

def get_word_reading_improved(node: fugashi.Node, dict_type: str) -> Optional[str]:
    """
    Extract reading from MeCab node - handles both UniDic and IPADIC
    
    Feature positions:
    IPADIC: [POS, POS1, POS2, POS3, Conjugation, Form, BaseForm, Reading, Pronunciation]
              0    1     2     3     4            5     6         7        8
    
    UniDic: [POS, POS1, POS2, POS3, POS4, POS5, POS6, BaseForm, Reading, ...]
             0    1     2     3     4     5     6     7          8
    """
    if not hasattr(node, 'feature') or not node.feature:
        return None
    
    features = node.feature
    
    try:
        if dict_type == "ipadic":
            # IPADIC: reading is at index 7, pronunciation at 8
            if len(features) > 7 and features[7] != '*':
                return features[7]
            # Fallback to surface if reading not available
            return None
            
        elif dict_type == "unidic":
            # UniDic: reading can be at different positions
            # Try position 8 first (kana reading)
            if len(features) > 8 and features[8] != '*':
                return features[8]
            # Try position 7 (base form reading)
            if len(features) > 7 and features[7] != '*':
                # Check if it's actually kana
                if re.match(r'^[\u3040-\u309F\u30A0-\u30FF]+$', features[7]):
                    return features[7]
            return None
        
        # Unknown dictionary type - search for kana in features
        for feat in features:
            if feat and feat != '*' and re.match(r'^[\u3040-\u309F\u30A0-\u30FF]+$', feat):
                return feat
        
        return None
        
    except Exception as e:
        logger.error(f"Error extracting reading: {e}")
        return None

def fix_particle_romaji_comprehensive(word: str, reading: str, romaji: str, pos: str = None) -> str:
    """
    Comprehensive particle and special word fixes
    """
    # Particles that need special romanization
    particle_map = {
        "は": "wa",      # Topic particle
        "へ": "e",       # Direction particle  
        "を": "wo",      # Object particle (can also be "o" but "wo" is more clear)
        "ず": "zu",      # Negative auxiliary
        "づ": "zu",      # Special zu
        "ぢ": "ji",      # Special ji
    }
    
    # Check exact word match first
    if word in particle_map:
        return particle_map[word]
    
    # Common verb/adjective endings
    ending_fixes = {
        "です": "desu",
        "ます": "masu",
        "ません": "masen",
        "ました": "mashita",
        "でした": "deshita",
        "だ": "da",
        "ない": "nai",
        "たい": "tai",
    }
    
    if word in ending_fixes:
        return ending_fixes[word]
    
    # Fix common romanization issues
    romaji = romaji.replace("wo", "wo")  # Keep wo for を
    
    # Fix long vowels
    romaji = re.sub(r'ou\b', 'ō', romaji)  # 行こう -> ikō
    romaji = re.sub(r'uu', 'ū', romaji)    # くうき -> kūki
    romaji = re.sub(r'ei', 'ei', romaji)   # Keep ei as is
    
    # Fix っ (small tsu) - should double next consonant
    # This should be handled by kakasi but verify
    
    return romaji

def convert_kana_to_romaji_improved(kana: str, kakasi_converter) -> str:
    """Improved kana to romaji conversion with post-processing"""
    if not kakasi_converter:
        return kana
    
    try:
        # Convert using PyKakasi
        romaji = kakasi_converter.do(kana)
        
        # Post-processing fixes
        # Fix common PyKakasi issues
        romaji = romaji.replace("'", "")  # Remove apostrophes
        
        # Fix spacing issues
        romaji = re.sub(r'\s+', ' ', romaji).strip()
        
        return romaji
        
    except Exception as e:
        logger.error(f"Kana conversion error: {e}")
        return kana

def mecab_to_romaji_perfect_v2(japanese: str, tagger, kakasi_converter, dict_type: str) -> str:
    """
    IMPROVED: Better handling of different word types and dictionary formats
    """
    if not tagger:
        if kakasi_converter:
            return kakasi_converter.do(japanese)
        return japanese
    
    try:
        romaji_parts = []
        
        for node in tagger(japanese):
            word = node.surface
            if not word:
                continue
            
            # Get POS info
            pos = None
            if hasattr(node, 'feature') and node.feature:
                pos = node.feature[0] if len(node.feature) > 0 else None
            
            # Get reading from MeCab
            reading = get_word_reading_improved(node, dict_type)
            
            # Convert to romaji
            if reading:
                # Use the kana reading for conversion
                romaji = convert_kana_to_romaji_improved(reading, kakasi_converter)
            elif kakasi_converter:
                # Fallback: convert surface form directly
                romaji = kakasi_converter.do(word)
            else:
                romaji = word
            
            # Apply comprehensive fixes
            romaji = fix_particle_romaji_comprehensive(word, reading or "", romaji, pos)
            
            # Special handling for numbers
            if re.match(r'^\d+$', word):
                romaji = word  # Keep numbers as-is
            
            # Special handling for English/Katakana words already in romaji
            if re.match(r'^[a-zA-Z]+$', word):
                romaji = word.lower()
            
            romaji_parts.append(romaji)
        
        # Join with spaces
        result = " ".join(romaji_parts)
        
        # Post-processing cleanup
        result = re.sub(r'\s+', ' ', result).strip()
        
        # Final particle fixes in context
        result = re.sub(r'\bha\b(?!\w)', 'wa', result)  # は as particle
        result = re.sub(r'\bhe\b(?=\s)', 'e', result)   # へ as particle
        
        logger.debug(f"Converted '{japanese}' -> '{result}' ({len(romaji_parts)} words)")
        
        return result
        
    except Exception as e:
        logger.error(f"MeCab conversion error: {e}")
        if kakasi_converter:
            return kakasi_converter.do(japanese)
        return japanese

def mecab_analyze_line_improved(japanese: str, tagger, kakasi_converter, dict_type: str) -> List[WordAnalysis]:
    """Improved detailed analysis"""
    if not tagger:
        return []
    
    try:
        analysis = []
        
        for node in tagger(japanese):
            word = node.surface
            if not word:
                continue
            
            # Get reading
            reading = get_word_reading_improved(node, dict_type)
            
            # Convert to romaji
            romaji = None
            if reading and kakasi_converter:
                romaji = convert_kana_to_romaji_improved(reading, kakasi_converter)
            elif kakasi_converter:
                romaji = kakasi_converter.do(word)
            
            # Get POS info
            pos = None
            pos_detail = None
            base_form = None
            
            if hasattr(node, 'feature') and node.feature:
                features = node.feature
                if len(features) > 0:
                    pos = features[0]
                if len(features) > 1:
                    pos_detail = features[1]
                
                # Base form position varies by dictionary
                if dict_type == "ipadic" and len(features) > 6:
                    base_form = features[6] if features[6] != '*' else None
                elif dict_type == "unidic" and len(features) > 7:
                    base_form = features[7] if features[7] != '*' else None
            
            # Apply fixes
            if romaji:
                romaji = fix_particle_romaji_comprehensive(word, reading or "", romaji, pos)
            
            analysis.append(WordAnalysis(
                surface=word,
                reading=reading,
                romaji=romaji,
                pos=pos,
                pos_detail=pos_detail,
                base_form=base_form,
                dict_type=dict_type
            ))
        
        return analysis
        
    except Exception as e:
        logger.error(f"MeCab analysis error: {e}")
        return []

# USAGE EXAMPLE:
"""
# In your setup_systems() function, replace:
tagger = initialize_mecab()

# With:
tagger, dict_type = initialize_mecab_improved()

# Then update your conversion functions to pass dict_type:
romaji = mecab_to_romaji_perfect_v2(japanese, tagger, kakasi_converter, dict_type)

# And for analysis:
analysis = mecab_analyze_line_improved(japanese, tagger, kakasi_converter, dict_type)
"""
