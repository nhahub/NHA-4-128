import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, Any
from app.configs import get_segmentation_model
from app.storage import save_segmentation_result, update_history
from app.core.preprocessing import preprocess_image


async def run_segmentation(request_id: str, image_filename: str, image_path: str):
    print(f" SEGMENTING {request_id}.......")

    await asyncio.sleep(1)

    seg_result: Dict[str, Any] = {
        "request_id": request_id,
        "image_filename": image_filename,
        "timestamp": datetime.now().isoformat(),
        "detections": []
    }

    # model loading in updated way !!!!!!
    model = get_segmentation_model()

    if model is None:
        seg_result["error"] = "Segmentation model not loaded"

    else:
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()

            img_array = preprocess_image(img_bytes, target_size=(256, 256))

            # prediction using the function of it
            masks = model.predict(img_array)

            seg_result.update({
                "model_used": True,
                "masks_shape": list(masks.shape),
                "max_confidence": float(np.max(masks))
            })

        except Exception as e:
            seg_result["error"] = str(e)

    seg_path = save_segmentation_result(seg_result, request_id, image_filename)

    update_history(
        request_id,
        segmentation_path=seg_path,
        status="segmented"
    )

    print(f"✅ SEGMENTATION DONE {request_id}")