import json
from pathlib import Path

import cv2
import numpy as np


def main() -> bool:
    rgb_dir = Path('preprocessed/rgb_256')
    i_dir = Path('preprocessed/I_channel')
    metadata_dir = Path('preprocessed/metadata')
    embedded_dir = Path('preprocessed/embedded_I_channel')
    embedded_preview_dir = Path('preprocessed/embedded_preview')
    process_collage_dir = Path('preprocessed/process_collage')

    rgb_files = sorted(rgb_dir.glob('*.png'))
    if not rgb_files:
        print('No RGB files found in preprocessed/rgb_256')
        return False

    first_image = rgb_files[0]
    image_id = first_image.stem

    print(f"Testing with image ID: {image_id}")
    print('=' * 60)

    img = cv2.imread(str(first_image))
    if img is None:
        print(f"Failed to read RGB image: {first_image}")
        return False
    print(f"RGB shape: {img.shape}")

    i_path = i_dir / f'{image_id}.npy'
    if not i_path.exists():
        print(f"Missing I-channel: {i_path}")
        return False

    i_channel = np.load(i_path)
    print(f"I-channel shape: {i_channel.shape}")
    print(f"I-channel range: [{i_channel.min():.4f}, {i_channel.max():.4f}]")

    if i_channel.shape != (256, 256):
        print('I-channel shape mismatch (expected 256x256)')
        return False

    if float(i_channel.min()) < 0.0 or float(i_channel.max()) > 1.0:
        print('I-channel out of normalized bounds [0,1]')
        return False

    metadata_path = metadata_dir / f'{image_id}.json'
    if not metadata_path.exists():
        print(f"Missing metadata: {metadata_path}")
        return False

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print('\nImage Metadata:')
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    embedded_path = embedded_dir / f'{image_id}.npy'
    if embedded_path.exists():
        embedded_i = np.load(embedded_path)
        print(f"\nEmbedded I-channel shape: {embedded_i.shape}")
        print(f"Embedded I-channel range: [{embedded_i.min():.4f}, {embedded_i.max():.4f}]")

        if embedded_i.shape != (256, 256):
            print('Embedded I-channel shape mismatch (expected 256x256)')
            return False

        if float(embedded_i.min()) < 0.0 or float(embedded_i.max()) > 1.0:
            print('Embedded I-channel out of normalized bounds [0,1]')
            return False

        preview_path = embedded_preview_dir / f'{image_id}.png'
        if not preview_path.exists():
            print(f"Missing embedded preview: {preview_path}")
            return False

        preview_img = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
        if preview_img is None:
            print(f"Failed to read embedded preview: {preview_path}")
            return False

        if preview_img.ndim != 3 or preview_img.shape[2] != 3:
            print('Embedded preview is not a 3-channel color image')
            return False

        print(f"Embedded preview shape: {preview_img.shape}")

        collage_path = process_collage_dir / f'{image_id}.png'
        if not collage_path.exists():
            print(f"Missing process collage: {collage_path}")
            return False

        collage_img = cv2.imread(str(collage_path), cv2.IMREAD_COLOR)
        if collage_img is None:
            print(f"Failed to read process collage: {collage_path}")
            return False

        if collage_img.ndim != 3 or collage_img.shape[2] != 3:
            print('Process collage is not a 3-channel color image')
            return False

        print(f"Process collage shape: {collage_img.shape}")

        embedding_metadata_path = metadata_dir / f'embedding_{image_id}.json'
        if not embedding_metadata_path.exists():
            print(f"Missing embedding metadata: {embedding_metadata_path}")
            return False

        with open(embedding_metadata_path, 'r') as f:
            embedding_metadata = json.load(f)

        required_keys = {
            'image_id', 'watermark_id', 'acm_iterations', 'catalan_iterations',
            'catalan_key', 'mosaic_shape', 'mosaic_grid', 'embedding_alpha',
            'embedded_i_path', 'host_i_min', 'host_i_max', 'embedded_i_min', 'embedded_i_max'
        }
        missing = required_keys - set(embedding_metadata.keys())
        if missing:
            print(f"Embedding metadata missing keys: {sorted(missing)}")
            return False

        print('\nEmbedding Metadata:')
        for key, value in embedding_metadata.items():
            print(f"  {key}: {value}")
    else:
        print('\nNo embedded I-channel found for this sample (embedding phase may not have run yet).')

    print('\nVerification complete: PASS')
    return True


if __name__ == '__main__':
    ok = main()
    if not ok:
        raise SystemExit(1)