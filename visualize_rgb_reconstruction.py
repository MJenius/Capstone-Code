import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def bgr_to_yiq(bgr: np.ndarray) -> np.ndarray:
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)

    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.275 * g - 0.321 * b
    q = 0.212 * r - 0.523 * g + 0.311 * b

    return np.stack([y, i, q], axis=2)


def yiq_to_bgr(yiq: np.ndarray) -> np.ndarray:
    y = yiq[:, :, 0].astype(np.float32)
    i = yiq[:, :, 1].astype(np.float32)
    q = yiq[:, :, 2].astype(np.float32)

    r = y + 0.956 * i + 0.621 * q
    g = y - 0.272 * i - 0.647 * q
    b = y - 1.106 * i + 1.703 * q

    bgr = np.stack([b, g, r], axis=2)
    return np.clip(bgr, 0, 255).astype(np.uint8)


def build_collage(original_bgr: np.ndarray, embedded_bgr: np.ndarray) -> np.ndarray:
    diff = np.abs(embedded_bgr.astype(np.float32) - original_bgr.astype(np.float32))
    diff_mag = diff.mean(axis=2)

    # Amplify the tiny embedding signal to make the map visible.
    diff_u8 = np.clip(diff_mag * 8.0, 0, 255).astype(np.uint8)
    diff_heat = cv2.applyColorMap(diff_u8, cv2.COLORMAP_TURBO)

    panels = [
        ("Original RGB", original_bgr),
        ("Reconstructed Embedded RGB", embedded_bgr),
        ("Change Map (|dRGB| mean x8)", diff_heat),
    ]

    cell_w, cell_h = 256, 256
    title_h = 32
    canvas = np.full((title_h + cell_h, cell_w * len(panels), 3), 245, dtype=np.uint8)

    for idx, (title, img) in enumerate(panels):
        x0 = idx * cell_w
        img_resized = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        canvas[title_h:title_h + cell_h, x0:x0 + cell_w] = img_resized
        cv2.putText(canvas, title, (x0 + 6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)

    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct embedded RGB and generate a change map.")
    parser.add_argument("--image-id", type=str, default=None, help="Image ID like 0001_00000")
    args = parser.parse_args()

    base = Path.cwd()
    rgb_dir = base / "preprocessed" / "rgb_256"
    embedded_i_dir = base / "preprocessed" / "embedded_I_channel"
    metadata_dir = base / "preprocessed" / "metadata"
    preview_dir = base / "preprocessed" / "embedded_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    if args.image_id is None:
        rgb_files = sorted(rgb_dir.glob("*.png"))
        if not rgb_files:
            print("No RGB files found in preprocessed/rgb_256")
            return 1
        image_id = rgb_files[0].stem
    else:
        image_id = args.image_id

    rgb_path = rgb_dir / f"{image_id}.png"
    embedded_i_path = embedded_i_dir / f"{image_id}.npy"
    metadata_path = metadata_dir / f"{image_id}.json"

    if not rgb_path.exists() or not embedded_i_path.exists() or not metadata_path.exists():
        print("Missing required files for image id:", image_id)
        print("Need:")
        print(" -", rgb_path)
        print(" -", embedded_i_path)
        print(" -", metadata_path)
        return 1

    original_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if original_bgr is None:
        print("Failed to read:", rgb_path)
        return 1

    embedded_i_norm = np.load(embedded_i_path).astype(np.float32)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    i_min = float(meta["i_channel_min"])
    i_max = float(meta["i_channel_max"])

    yiq_original = bgr_to_yiq(original_bgr)
    embedded_i = embedded_i_norm * (i_max - i_min) + i_min

    yiq_embedded = yiq_original.copy()
    yiq_embedded[:, :, 1] = embedded_i

    embedded_bgr = yiq_to_bgr(yiq_embedded)

    reconstructed_path = preview_dir / f"reconstructed_embedded_rgb_{image_id}.png"
    cv2.imwrite(str(reconstructed_path), embedded_bgr)

    collage = build_collage(original_bgr, embedded_bgr)
    collage_path = preview_dir / f"process_demo_color_{image_id}.png"
    cv2.imwrite(str(collage_path), collage)

    print(f"Image ID: {image_id}")
    print(f"Saved reconstructed RGB: {reconstructed_path}")
    print(f"Saved collage: {collage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
