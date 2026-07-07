"""
image_downloader.py
===================
Manages downloading, caching, and validating medical images from URLs.

Features:
- Hash-based deduplication (avoid re-downloading same image)
- Network retry logic with exponential backoff
- MIME type validation (ensure valid image)
- File size validation
- Graceful error handling with logging
"""

import logging
import hashlib
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = Path(__file__).parent / "data" / "downloaded_images"
CACHE_METADATA_FILE = CACHE_DIR / ".cache_manifest.json"
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # exponential backoff multiplier
TIMEOUT = 10  # seconds
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
VALID_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def get_image_cache_dir() -> Path:
    """Create and return the image cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def hash_url(url: str) -> str:
    """Generate SHA256 hash of URL for deduplication."""
    return hashlib.sha256(url.encode()).hexdigest()


def get_cached_path(url: str) -> Path:
    """Get the cache file path for a given URL."""
    url_hash = hash_url(url)
    # Try to infer extension from URL
    try:
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix.lower() or ".jpg"
    except Exception:
        ext = ".jpg"
    
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        ext = ".jpg"
    
    return get_image_cache_dir() / f"{url_hash}{ext}"


def image_exists_locally(url: str) -> bool:
    """Check if image is already cached."""
    cache_path = get_cached_path(url)
    exists = cache_path.exists()
    if exists:
        logger.debug(f"Image cache hit: {cache_path}")
    return exists


def validate_image_response(response: requests.Response) -> tuple[bool, str]:
    """Validate HTTP response before saving as image."""
    # Check status code
    if response.status_code != 200:
        return False, f"HTTP {response.status_code}"
    
    # Check MIME type
    content_type = response.headers.get("content-type", "").lower()
    if not any(mime in content_type for mime in VALID_MIME_TYPES):
        return False, f"Invalid MIME type: {content_type}"
    
    # Check content length
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
            if size > MAX_FILE_SIZE:
                return False, f"File too large: {size} bytes"
        except ValueError:
            pass
    
    return True, "OK"


def validate_image_file(file_path: Path) -> bool:
    """Validate downloaded image file integrity."""
    try:
        # Try to open with PIL to ensure valid image
        with Image.open(file_path) as img:
            img.verify()
        logger.debug(f"Image file validated: {file_path}")
        return True
    except Exception as e:
        logger.warning(f"Image validation failed: {e}")
        return False


def download_image(url: str, force_refresh: bool = False) -> Optional[Path]:
    """
    Download image from URL with retry logic and caching.
    
    Args:
        url: Direct download URL for the image
        force_refresh: If True, ignore cache and re-download
    
    Returns:
        Path to downloaded/cached image, or None if failed
    """
    if not url or not isinstance(url, str):
        logger.warning("Invalid URL provided")
        return None
    
    # Check cache first
    cache_path = get_cached_path(url)
    if not force_refresh and image_exists_locally(url):
        return cache_path
    
    # Create cache directory
    get_image_cache_dir()
    
    # Attempt download with retry
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Downloading image (attempt {attempt + 1}/{MAX_RETRIES}): {url}")
            
            response = requests.get(
                url,
                timeout=TIMEOUT,
                allow_redirects=True,
                headers={"User-Agent": "DermaScan-AI/1.0"}
            )
            
            # Validate response
            is_valid, reason = validate_image_response(response)
            if not is_valid:
                logger.warning(f"Invalid response: {reason}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_BACKOFF ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None
            
            # Save to cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(response.content)
            logger.info(f"Image downloaded and cached: {cache_path}")
            
            # Validate downloaded file
            if not validate_image_file(cache_path):
                logger.error(f"Downloaded file validation failed")
                cache_path.unlink(missing_ok=True)
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_BACKOFF ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return None
            
            return cache_path
        
        except requests.Timeout:
            logger.warning(f"Timeout downloading image (attempt {attempt + 1})")
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_BACKOFF ** attempt
                time.sleep(wait_time)
        
        except requests.RequestException as e:
            logger.warning(f"Request error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_BACKOFF ** attempt
                time.sleep(wait_time)
        
        except Exception as e:
            logger.error(f"Unexpected error downloading image: {e}")
            return None
    
    logger.error(f"Failed to download image after {MAX_RETRIES} attempts")
    return None


def get_or_download_image(url: str) -> Optional[Path]:
    """
    Smart wrapper: return cached image if exists, otherwise download.
    
    Args:
        url: Direct download URL for the image
    
    Returns:
        Path to image file, or None if unavailable
    """
    if not url:
        logger.warning("Empty URL provided to get_or_download_image")
        return None
    
    # Return cached if available
    if image_exists_locally(url):
        return get_cached_path(url)
    
    # Otherwise download
    return download_image(url, force_refresh=False)


def clear_cache() -> None:
    """Clear all cached images (use cautiously)."""
    try:
        if CACHE_DIR.exists():
            for file in CACHE_DIR.glob("*"):
                if file.is_file() and not file.name.startswith("."):
                    file.unlink()
            logger.info(f"Cache cleared: {CACHE_DIR}")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")


def get_cache_stats() -> dict:
    """Get cache statistics."""
    if not CACHE_DIR.exists():
        return {"total_files": 0, "total_size_mb": 0}
    
    files = list(CACHE_DIR.glob("*"))
    image_files = [f for f in files if f.is_file() and not f.name.startswith(".")]
    total_size = sum(f.stat().st_size for f in image_files)
    
    return {
        "total_files": len(image_files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "cache_dir": str(CACHE_DIR)
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test: Print cache stats
    print("Cache stats:", get_cache_stats())
    
    # Test: Try downloading a test image (replace with real URL)
    # test_url = "https://example.com/test.jpg"
    # result = get_or_download_image(test_url)
    # print(f"Result: {result}")
