import io
from PIL import Image

def is_valid_image(file_bytes: bytes) -> bool:
    try:
        Image.open(io.BytesIO(file_bytes)).verify()
        return True
    except:
        return False
