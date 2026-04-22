"""
Binary watermark generation utility.

Generates a binary (black and white) watermark image from text for use in
the watermarking pipeline. Outputs both PNG (for visualization) and NPY
(for numerical processing).
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def generate_binary_watermark(text="PW26_PAC_01", size=(32, 32), output_path="data/watermark/"):
    """
    Generates a 32x32 binary watermark from a text string.
    
    Args:
        text: Text to render as watermark (default: "PW26_PAC_01")
        size: Target watermark size as (width, height) tuple (default: (32, 32))
        output_path: Directory to save watermark files (default: "data/watermark/")
        
    Returns:
        Watermark as numpy array (binary: 0 or 1 values)
    """
    # 1. Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # 2. Create a new black image (mode '1' for 1-bit pixels)
    # Use a large canvas for better font rendering quality before downsampling.
    canvas_size = 256
    img = Image.new('1', (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)

    # 3. Load a font — use size 32 so text is legible after downsampling to 32x32
    try:
        # Attempt to use a standard system font
        font = ImageFont.truetype("arial.ttf", 32)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 32)
        except IOError:
            # Fallback: PIL default is tiny (~8px). Log a prominent warning.
            import logging
            logging.warning(
                "generate_watermark: No TrueType font found. Falling back to PIL default "
                "(~8px). The 32x32 watermark may be blank or illegible. "
                "Install 'arial.ttf' or 'DejaVuSans.ttf' for reliable results."
            )
            font = ImageFont.load_default()

    # 4. Calculate text position (centered)
    # Use textbbox for newer Pillow versions; textsize is deprecated
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        text_width, text_height = draw.textsize(text, font=font)
        
    x = (canvas_size - text_width) // 2
    y = (canvas_size - text_height) // 2

    # 5. Draw the text in white
    draw.text((x, y), text, fill=1, font=font)

    # 6. Resize down to 32x32 using nearest neighbor to keep it binary
    img = img.resize(size, Image.NEAREST)

    # 7. Convert to NumPy array and save
    watermark_array = np.array(img).astype(np.uint8)
    
    # Save as image for visual check
    img.save(os.path.join(output_path, "watermark_binary.png"))
    # Save as .npy for the scrambling phase
    np.save(os.path.join(output_path, "watermark_binary.npy"), watermark_array)

    print(f"Success: 32x32 binary watermark for '{text}' saved to {output_path}")
    print(f"  - Visual: watermark_binary.png")
    print(f"  - Array: watermark_binary.npy")
    print(f"  - Shape: {watermark_array.shape}")
    print(f"  - Unique values: {np.unique(watermark_array)}")

    # Validate: warn loudly if the watermark is trivially blank or uniform.
    dominant = np.bincount(watermark_array.flatten()).max()
    fill_fraction = dominant / watermark_array.size
    if fill_fraction > 0.90:
        raise RuntimeError(
            f"generate_watermark: watermark appears blank or near-uniform "
            f"({fill_fraction*100:.1f}% of pixels share the same value). "
            "Check that a legible TrueType font is available or adjust canvas_size/font_size."
        )
    
    return watermark_array


if __name__ == "__main__":
    generate_binary_watermark()
