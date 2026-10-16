from PIL import Image

def resize_image_max_dim(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Resize PIL image keeping aspect ratio if any dimension exceeds max_dim."""
    width, height = img.size
    if max(width, height) > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        else:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img
