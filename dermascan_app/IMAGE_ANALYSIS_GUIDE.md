# Image Analysis Integration - Implementation Guide

## Overview

This enhancement adds automated medical image download and vision model analysis to the DermaScan AI RAG pipeline. Doctors can now retrieve complete patient profiles that combine textual data, medical images, and AI-powered image analysis.

## New Modules

### 1. `image_downloader.py`
Manages downloading, caching, and validating medical images from URLs.

**Key Functions:**
- `get_or_download_image(url)` - Smart wrapper: returns cached image or downloads
- `download_image(url, force_refresh=False)` - Downloads with retry logic and validation
- `image_exists_locally(url)` - Check cache status
- `get_cache_stats()` - Monitor cache usage
- `clear_cache()` - Clear all cached images (careful use)

**Features:**
- Hash-based URL deduplication (SHA256)
- Network retry with exponential backoff (3 attempts)
- MIME type validation (jpeg, png, webp, gif only)
- File size validation (max 50MB)
- Automatic timeout handling (10 second timeout)
- Cache directory: `Data/downloaded_images/`

**Error Handling:**
- Invalid URL format → Returns None
- Network failures → Auto-retry with backoff
- Corrupted downloads → PIL validation + file deletion
- Missing URLs → Graceful degradation

### 2. `vision_analyzer.py`
Analyzes skin lesion images using Groq's vision model (llama-3.2-90b-vision-preview).

**Key Functions:**
- `analyze_image_with_vision(image_path, patient_context)` - Call vision API
- `parse_vision_response(response_text)` - Extract numeric metrics
- `analyze_and_format(image_path, patient_context)` - Complete pipeline

**Vision Prompt:**
Analyzes:
- Lesion color uniformity & variation
- Border symmetry and irregularity
- Texture and surface patterns
- Asymmetry, border, and color scores (0-10 scale)
- Diameter estimation
- Diagnostic feature assessment
- Image quality evaluation

**Output:**
Structured analysis with:
- Raw AI interpretation
- Extracted metrics (asymmetry_score, border_irregularity_score, color_variation_score, diameter_mm)
- Formatted context suitable for LLM processing

### 3. Modified `agent.py`
Added new LangChain tool: `lookup_patient_with_image(patient_id)`

**New Tool:**
```python
@tool
def lookup_patient_with_image(patient_id: str) -> str:
    """[DOCTOR MODE ONLY] Retrieve complete patient record including 
    downloaded medical image analysis."""
```

**Workflow:**
1. Verify doctor role (restricted access)
2. Fetch patient record from Excel
3. Extract `image_url` field
4. Download image (with caching)
5. Analyze with vision model
6. Merge: vision analysis + patient text data
7. Return combined context

**Integration:**
- Added to agent's tools list in `build_agent()`
- Documented in SYSTEM_PROMPT
- Doctor-only access control enforced

## Excel Schema Requirement

For full functionality, the `updated_file_2.xlsx` file must include an `image_url` column in the Patients sheet:

```
Patients sheet columns:
- patient_id (e.g., "P0001")
- name
- age
- gender
- country
- diagnosis
- biopsy_result
- lesion_size_mm
- lesion_color
- lesion_location
- border_irregularity
- asymmetry
- skin_type_fitzpatrick
- UV_exposure_level
- genetic_mutation
- hereditary_risk_score
- family_history_skin_cancer
- immunosuppressed
- family_id
- image_url (NEW) ← Direct download link for medical image
```

**Image URL Requirements:**
- Must be publicly accessible (no authentication)
- Should be a direct link (no redirects recommended)
- Valid image formats: JPEG, PNG, WebP, GIF
- File size: < 50 MB

## Usage Examples

### Doctor Mode Chat
```
User (Doctor): "lookup_patient_with_image P0042"

Agent Response:
📋 COMPLETE PATIENT PROFILE — P0042

🔬 AI VISION ANALYSIS OF MEDICAL IMAGE:

**Lesion Characteristics**:
- Color: Mixed brown and black with asymmetry: 7/10
- Border: Highly irregular, definition: 3/10
- Diameter: approximately 8-10mm

[... detailed vision analysis ...]

**TEXTUAL PATIENT DATA:**

Patient P0042 — Ahmed Salem, 45 y/o Male, Egypt
Diagnosis: Melanoma | Biopsy: Positive
Lesion: 9mm, dark brown, right shoulder...
[... full patient record ...]
```

### Graceful Degradation
- No image URL → Returns text patient data only
- Download fails → Returns text patient data only
- Vision analysis fails → Returns text + image cache confirmation
- Patient not found → Clear error message

## Cache Management

Cache stored in: `dermascan_app/data/downloaded_images/`

**Cache Files:**
- Named by URL hash: `{SHA256(url)}.{ext}`
- Includes file extension inference from URL

**Cache Stats:**
```python
from image_downloader import get_cache_stats
stats = get_cache_stats()
# Returns: {"total_files": 5, "total_size_mb": 23.4, "cache_dir": "..."}
```

**Clear Cache (if needed):**
```python
from image_downloader import clear_cache
clear_cache()  # Removes all cached images
```

## Error Handling & Logging

All modules use Python's standard logging module:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Log Levels:**
- DEBUG: Cache hits, download progress, API calls
- INFO: Successful operations (download complete, analysis done)
- WARNING: Retries, validation issues, missing URLs
- ERROR: Critical failures (network errors after retries, API failures)

**Key Log Messages:**
- `"Image cache hit: ..."` - Image served from cache
- `"Downloading image (attempt X/3): ..."` - Download started
- `"Image validation failed: ..."` - Downloaded file corrupted
- `"Vision analysis completed successfully"` - Analysis done
- `"Could not initialize vision client"` - Vision API unavailable

## Performance Considerations

### Download Optimization
- **Caching**: Avoids re-downloading same image
- **URL hashing**: O(1) lookup for cache existence
- **Timeout**: 10 seconds per download request
- **Retry backoff**: 2^attempt seconds (2s, 4s, 8s max)

### Vision Analysis Cost
- **Model**: llama-3.2-90b-vision-preview (Groq)
- **Input tokens**: ~200-300 per image + patient context
- **Output tokens**: ~300-500 for detailed analysis
- **Temperature**: 0.3 (deterministic clinical analysis)

### Database Queries
- Patient lookup: O(n) pandas dataframe search (cached in memory)
- Family lookup: O(n) dataframe search (same as existing)
- No additional database calls beyond existing architecture

## Backward Compatibility

✅ **All existing functionality preserved:**
- Existing `lookup_patient()` tool unchanged
- Image upload classification/segmentation unchanged
- Image registry (perceptual hash matching) unchanged
- Patient booking workflow unchanged
- Knowledge base retrieval unchanged

**New tool addition:**
- Additive only: new tool available to agent
- Does not affect existing tools
- Doctor-only access (no patient exposure)
- Graceful degradation if image unavailable

## Dependencies

New dependency added to `requirements.txt`:
- `requests` - HTTP client for downloading images

Existing dependencies:
- `langchain`, `langchain-groq` - LLM integration
- `pillow` - Image processing & validation
- `pandas` - Excel data handling
- `tensorflow`, `opencv-python` - Existing CV models

## Testing the Integration

### Unit Test Example
```python
from pathlib import Path
from image_downloader import get_or_download_image, get_cache_stats
from vision_analyzer import analyze_and_format

# Test download & cache
test_url = "https://example.com/test_image.jpg"
image_path = get_or_download_image(test_url)
assert image_path.exists()

# Test cache stats
stats = get_cache_stats()
print(f"Cached files: {stats['total_files']}")

# Test vision analysis
analysis = analyze_and_format(image_path, "Patient context")
assert analysis is not None
assert "AI VISION ANALYSIS" in analysis
```

### Integration Test
1. Set `GROQ_API_KEY` environment variable
2. Start app: `uvicorn main:app --reload --port 8000`
3. In chat (Doctor mode): `lookup_patient_with_image P0042`
4. Verify:
   - Image downloads to cache
   - Vision analysis appears in response
   - Patient data merged correctly
   - No errors in logs

## Future Enhancements

Possible additions:
1. **Batch image analysis** - Process multiple patient images
2. **Image comparison** - Compare previous vs. current lesion images
3. **Alternative vision models** - GPT-4V, Claude Vision, local models
4. **Image storage database** - Archive analyzed images with metadata
5. **CAM visualization** - Show activation maps from vision model
6. **Confidence scoring** - Numerical benign/malignant probability
7. **Trend analysis** - Track lesion changes over time
8. **Multi-image support** - Multiple lesions per patient

## Troubleshooting

**Issue: "Vision client not available"**
- Check: `GROQ_API_KEY` environment variable set
- Check: Groq API key valid and not expired
- Check: Network connectivity to Groq API

**Issue: "Could not download image"**
- Check: URL is publicly accessible (no authentication)
- Check: URL is not a redirect (direct link preferred)
- Check: File size < 50MB
- Check: Network connectivity
- Check: Internet not behind restrictive firewall

**Issue: Image in cache but not updated**
- Use `force_refresh=True` in `download_image()`
- Or: Call `clear_cache()` to reset

**Issue: Low vision analysis quality**
- Check: Image quality (resolution, focus, lighting)
- Verify: Image file not corrupted
- Consider: Patient context (helps model understanding)

## Support & Questions

For issues or questions:
1. Check logs: `logging.basicConfig(level=logging.DEBUG)`
2. Test modules independently
3. Verify Excel schema includes `image_url` column
4. Confirm network/firewall allows downloads
5. Ensure Groq API key valid
