"""
Utility functions for the EDA.
"""

import hashlib
from PIL import Image
import imagehash


def get_md5_hash(image_path):
    """
    Generate an MD5 hash for an image file.
    Useful for detecting exact duplicate images.
    """
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_perceptual_hash(image_path):
    """
    Generate perceptual hash (pHash) for an image.

    pHash captures visual similarity between images.
    Images with small hash distance are visually similar.
    """
    try:
        img = Image.open(image_path)
        return imagehash.phash(img)
    except Exception:
        return None

'''
def get_all_hashes(image_path):
    """
    Generate multiple perceptual hashes for an image:
    - phash: structure-based hash
    - dhash: gradient-based hash
    - ahash: average color/brightness hash
    - whash: wavelet-based hash

    Returns a dictionary with hash type as keys.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        hashes = {
            'phash': imagehash.phash(img),
            'dhash': imagehash.dhash(img),
            'whash': imagehash.whash(img)
        }
        return hashes
    except Exception:
        return None
'''

def get_all_hashes(image_path):
    """
    Generate perceptual hashes for an image after converting it to grayscale.
    """

    try:
        img = Image.open(image_path)

        # Convert to grayscale
        img = img.convert("L")

        hashes = {
            "phash": imagehash.phash(img),
            "dhash": imagehash.dhash(img),
            "whash": imagehash.whash(img)
        }

        return hashes

    except Exception:
        return None