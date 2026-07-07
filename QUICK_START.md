# Quick Start: Testing Image Analysis Integration

## Pre-Test Setup (5 minutes)

### 1. Install New Dependency
```bash
cd dermascan_app
pip install requests
# Or: pip install -r requirements.txt
```

### 2. Prepare Excel File
Your `data/updated_file_2.xlsx` Patients sheet needs an `image_url` column:

**Add Column:**
- Column name: `image_url`
- Column type: Text/String
- Content: Direct download URLs to medical images

**Example rows:**
```
patient_id | name          | age | diagnosis   | ... | image_url
P0001      | Ahmed Salem   | 45  | Melanoma    | ... | https://example.com/p0001.jpg
P0042      | Fatima Karim  | 38  | Nevus       | ... | https://example.com/p0042.png
P0123      | Omar Hassan   | 52  | Carcinoma   | ... | https://example.com/p0123.jpg
```

**Image URL Requirements:**
- Must be publicly accessible (no login)
- Must be direct link (no redirects)
- Formats: JPEG, PNG, WebP, GIF
- Size: < 50 MB each

### 3. Verify Environment
```bash
# Check API key set
echo $env:GROQ_API_KEY  # Windows PowerShell
# # Should show: your_groq_api_key_here
# Or export if not set:
$env:GROQ_API_KEY="your_groq_api_key"
```

---

## Test 1: Direct Module Testing (Python REPL)

```python
# Test image downloading
from image_downloader import get_or_download_image, get_cache_stats

url = "https://example.com/test_image.jpg"
image_path = get_or_download_image(url)
print(f"Downloaded to: {image_path}")
print(f"File exists: {image_path.exists()}")

# Check cache
stats = get_cache_stats()
print(f"Cache: {stats}")

# Try downloading same URL again (should use cache)
image_path2 = get_or_download_image(url)
print(f"Same image: {image_path == image_path2}")  # Should be True
```

```python
# Test vision analysis
from vision_analyzer import analyze_and_format
from pathlib import Path

image_path = Path("data/downloaded_images/...")  # From test above
patient_context = "45 year old male, family history of melanoma"

analysis = analyze_and_format(image_path, patient_context)
if analysis:
    print("✅ Vision analysis successful:")
    print(analysis[:500] + "...")
else:
    print("❌ Vision analysis failed")
```

---

## Test 2: Full Integration Test (Chat Interface)

### Start the Application
```bash
cd dermascan_app
uvicorn main:app --reload --port 8000
```

### In Browser
Navigate to: `http://localhost:8000`

### Test Steps
1. **Set Role**: Click "Doctor" button (top-right)
2. **Send Query**: Type in chat:
   ```
   lookup_patient_with_image P0042
   ```
3. **Expected Response** (within 10-15 seconds):
   ```
   📋 COMPLETE PATIENT PROFILE — P0042

   🔬 AI VISION ANALYSIS OF MEDICAL IMAGE:

   **Lesion Characteristics**:
   - Color: Mixed brown tones, variation: 7/10
   - Border: Irregular, regularity: 3/10
   - Asymmetry: 6/10
   - Diameter: ~8-10mm
   
   [... detailed analysis ...]

   **TEXTUAL PATIENT DATA:**

   Patient P0042 — Fatima Karim, 38 y/o Female, Egypt
   [... patient record ...]
   ```

### Expected Behaviors
- ✅ First request: Downloads image + analyzes (may take 15-20 seconds)
- ✅ Second request for same patient: Much faster (uses cache)
- ✅ Missing patient ID: Clear error message
- ✅ No image URL: Returns patient text only (graceful degradation)
- ✅ Download fails: Returns patient text only (no crash)

---

## Test 3: Logging & Debug Output

### Enable Debug Logging
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
```

### Watch For These Messages
```
[image_downloader] DEBUG: Image cache hit: data/downloaded_images/abc123.jpg
[image_downloader] INFO: Downloading image (attempt 1/3): https://...
[image_downloader] DEBUG: Image file validated: data/downloaded_images/abc123.jpg
[vision_analyzer] INFO: Analyzing image with vision model: abc123.jpg
[vision_analyzer] INFO: Vision analysis completed successfully
[agent] DEBUG: lookup_patient_with_image invoked for P0042
[agent] INFO: Vision analysis completed for P0042
```

---

## Test 4: Cache Management

### Check Cache Status
```python
from image_downloader import get_cache_stats, clear_cache

stats = get_cache_stats()
print(f"Cached images: {stats['total_files']}")
print(f"Cache size: {stats['total_size_mb']} MB")
```

### Clear Cache (if needed)
```python
from image_downloader import clear_cache

clear_cache()
print("Cache cleared")
```

---

## Test 5: Graceful Degradation

### Test A: Missing Image URL
```
Chat: "lookup_patient_with_image P0999"
Expected: Patient text data only (no crash)
```

### Test B: Invalid Image URL
```
# Add patient with bad URL
image_url: "https://invalid-domain-12345.com/image.jpg"

Chat: "lookup_patient_with_image P0999"
Expected: After 3 retries, returns patient text only
```

### Test C: Network Offline
```bash
# Disconnect network, then:
Chat: "lookup_patient_with_image P0042"
Expected: Graceful error, returns patient text only
```

---

## Troubleshooting

### Issue: "Vision client not available"
```
Solution:
1. Check GROQ_API_KEY: echo $env:GROQ_API_KEY
2. Verify key is valid in Groq console
3. Test network: ping api.groq.com
```

### Issue: Image download stuck
```
Solution:
1. Check URL is accessible: curl https://...
2. Check file size < 50MB
3. Check network connectivity
4. Images timeout after 10 seconds (retry auto-activates)
```

### Issue: Cache not updating
```
Solution:
from image_downloader import download_image
image_path = download_image(url, force_refresh=True)
```

### Issue: "Corrupted images" errors
```
Solution:
1. Check source image is valid (try downloading manually)
2. Verify MIME type (should be image/jpeg, image/png, etc.)
3. Try different URL if available
```

---

## Performance Expectations

### First Request (P0042)
- Download: 3-8 seconds (depends on image size & network)
- Vision analysis: 8-12 seconds
- Total: 15-20 seconds
- Status: ⏳ Please wait

### Second Request (Same P0042)
- Cache lookup: < 100ms
- Vision analysis: 8-12 seconds (API call still needed)
- Total: ~10 seconds
- Status: ⚡ Faster (image cached)

### Third Request (Different P0043)
- Download: 3-8 seconds
- Vision analysis: 8-12 seconds
- Total: 15-20 seconds
- Status: ⏳ New image

---

## Next Steps After Testing

1. ✅ Verify all tests pass
2. ✅ Monitor logs for errors
3. ✅ Check cache is working (reused images should be faster)
4. ✅ Validate vision analysis quality
5. ✅ Test with multiple patients
6. ✅ Review output formatting in chat

Then:
- Deploy to production
- Monitor API usage & costs
- Set up cache cleanup schedule (optional)
- Gather feedback from doctors

---

## Questions?

Refer to:
1. **IMAGE_ANALYSIS_GUIDE.md** — Detailed documentation
2. **IMPLEMENTATION_SUMMARY.md** — Architecture & design
3. **Logs** — Debug info at DEBUG level
4. **Code comments** — In image_downloader.py & vision_analyzer.py

Good luck! 🚀
