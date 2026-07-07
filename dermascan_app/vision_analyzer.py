"""
vision_analyzer.py
==================
Analyzes skin lesion images using Groq's vision model.

Provides:
- Vision model API calls with error handling
- Response parsing for clinical insights
- Context formatting for RAG pipeline
- Fallback to text-only if vision unavailable
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import base64
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Configure logging
logger = logging.getLogger(__name__)

# Vision model configuration
# Updated: llama-3.2-11b-vision-preview is decommissioned, using meta-llama/llama-4-scout-17b-16e-instruct
VISION_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"  # Groq's current vision model
TEMPERATURE = 0.3


def get_vision_client() -> ChatGroq:
    """Initialize Groq client for vision API."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set")
        return None
    
    try:
        return ChatGroq(
            model_name=VISION_MODEL_NAME,
            temperature=TEMPERATURE,
            api_key=api_key,
            request_timeout=120,
        )
    except Exception as e:
        logger.error(f"Failed to initialize vision client: {e}")
        return None


def image_to_base64(image_path: Path) -> str:
    """Convert image file to base64 string."""
    try:
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        return ""


def get_image_media_type(image_path: Path) -> str:
    """Infer media type from file extension."""
    ext = image_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return mime_map.get(ext, "image/jpeg")


def analyze_image_with_vision(
    image_path: Path,
    patient_context: str = ""
) -> Optional[str]:
    """
    Analyze skin lesion image using Groq vision model.
    
    Args:
        image_path: Path to downloaded image file
        patient_context: Existing patient data/diagnosis for context
    
    Returns:
        Vision analysis as text, or None if analysis failed
    """
    if not image_path or not image_path.exists():
        logger.warning(f"Image file not found: {image_path}")
        return None
    
    # Initialize client
    client = get_vision_client()
    if not client:
        logger.warning("Vision client not available")
        return None
    
    try:
        # Encode image
        b64_image = image_to_base64(image_path)
        if not b64_image:
            logger.error("Failed to encode image to base64")
            return None
        
        media_type = get_image_media_type(image_path)
        
        # Build prompt with patient context if available
        context_note = ""
        if patient_context:
            context_note = f"\n\nExisting patient data:\n{patient_context}\n"
        
        prompt = f"""You are an expert dermatopathologist analyzing a skin lesion image.
Provide a detailed clinical analysis focusing on:

1. **Lesion Characteristics**:
   - Color uniformity and specific colors present
   - Border definition and symmetry
   - Texture and surface patterns
   - Overall shape and size estimation

2. **Key Diagnostic Features**:
   - Asymmetry index (0-10 scale)
   - Border regularity (0-10 scale)
   - Color variation (0-10 scale)
   - Diameter estimation in mm (if visible)

3. **Pattern Analysis** (if visible):
   - Dots or globules
   - Streaks or radiating structures
   - Blue-white veil
   - Pigment network
   - Other notable patterns

4. **Risk Assessment**:
   - Concerning features present
   - Benign vs malignant likelihood indicators
   - Recommended next steps

5. **Image Quality**:
   - Lighting and focus quality
   - Coverage and visibility

Please be concise and specific. Use numeric scales where indicated.
Format the response clearly with section headers.{context_note}"""
        
        # Call vision API
        logger.info(f"Analyzing image with vision model: {image_path.name}")
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{b64_image}"
                    }
                }
            ]
        )
        
        response = client.invoke([message])
        analysis_text = response.content
        
        logger.info("Vision analysis completed successfully")
        return analysis_text
    
    except Exception as e:
        logger.error(f"Vision analysis failed: {e}")
        return None


def parse_vision_response(response_text: str) -> dict:
    """
    Parse vision model response to extract structured data.
    
    Args:
        response_text: Raw response from vision model
    
    Returns:
        Dictionary with extracted metrics and observations
    """
    if not response_text:
        return {}
    
    # Simple parsing: look for numeric patterns and key phrases
    parsed = {
        "raw_analysis": response_text,
        "sections": {},
    }
    
    # Split by common section headers
    sections = response_text.split("**")
    current_section = "general"
    
    for i, section in enumerate(sections):
        if i % 2 == 1:  # Odd indices are section titles (between **)
            current_section = section.lower().replace(" ", "_")
            parsed["sections"][current_section] = ""
        else:
            if current_section not in parsed["sections"]:
                parsed["sections"][current_section] = ""
            parsed["sections"][current_section] += section
    
    # Try to extract numeric scales (simple regex-based)
    import re
    
    # Look for patterns like "asymmetry: 7/10" or "border: 3 out of 10"
    asymmetry_match = re.search(r"asymmetr[^:]*:?\s*(\d+)\s*(?:/|out of)\s*10", response_text, re.IGNORECASE)
    border_match = re.search(r"border[^:]*:?\s*(\d+)\s*(?:/|out of)\s*10", response_text, re.IGNORECASE)
    color_match = re.search(r"color[^:]*:?\s*(\d+)\s*(?:/|out of)\s*10", response_text, re.IGNORECASE)
    diameter_match = re.search(r"(?:diameter|size)[^:]*:?\s*(\d+)\s*mm", response_text, re.IGNORECASE)
    
    if asymmetry_match:
        parsed["asymmetry_score"] = int(asymmetry_match.group(1))
    if border_match:
        parsed["border_irregularity_score"] = int(border_match.group(1))
    if color_match:
        parsed["color_variation_score"] = int(color_match.group(1))
    if diameter_match:
        parsed["diameter_mm"] = int(diameter_match.group(1))
    
    return parsed


def format_vision_context(
    vision_analysis: str,
    parsed_metrics: dict = None
) -> str:
    """
    Format vision analysis for inclusion in RAG context.
    
    Args:
        vision_analysis: Raw analysis from vision model
        parsed_metrics: Optional parsed metrics dictionary
    
    Returns:
        Formatted text suitable for prepending to patient context
    """
    if not vision_analysis:
        return ""
    
    header = "**🔬 AI VISION ANALYSIS OF MEDICAL IMAGE:**\n\n"
    content = vision_analysis
    
    # Add metrics section if available
    if parsed_metrics:
        metrics_text = "\n\n**EXTRACTED METRICS:**\n"
        for key, value in parsed_metrics.items():
            if key not in ("raw_analysis", "sections") and not isinstance(value, dict):
                metrics_text += f"- {key}: {value}\n"
        content += metrics_text
    
    return header + content + "\n\n---\n"


def analyze_and_format(
    image_path: Path,
    patient_context: str = ""
) -> Optional[str]:
    """
    Complete pipeline: analyze image, parse, and format for RAG.
    
    Args:
        image_path: Path to image file
        patient_context: Existing patient data
    
    Returns:
        Formatted vision analysis or None if failed
    """
    # Get raw analysis
    analysis = analyze_image_with_vision(image_path, patient_context)
    if not analysis:
        return None
    
    # Parse metrics
    metrics = parse_vision_response(analysis)
    
    # Format for RAG
    return format_vision_context(analysis, metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test: Show available environment
    print(f"GROQ_API_KEY set: {'GROQ_API_KEY' in os.environ}")
    
    # Test: Parse sample response
    sample_response = """
**Lesion Characteristics**:
- Color: Mixed browns and blacks, asymmetry: 7/10
- Border: Irregular, border regularity: 3/10
- Diameter: 8mm

**Key Diagnostic Features**:
- Color variation: 8/10
- Multiple concerning features present
"""
    
    parsed = parse_vision_response(sample_response)
    print(f"\nParsed metrics: {json.dumps(parsed, indent=2)}")
