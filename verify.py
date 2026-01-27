import numpy as np
import cv2
import json
from pathlib import Path

# Find the first image in the preprocessed directory
rgb_dir = Path('preprocessed/rgb_256')
first_image = sorted(rgb_dir.glob('*.png'))[0]
image_id = first_image.stem  # Get filename without extension

print(f"Testing with image ID: {image_id}")
print("="*60)

# Check RGB image
img = cv2.imread(str(first_image))
print(f"RGB shape: {img.shape}")  # Should be (256, 256, 3)

# Check I-channel
i_channel = np.load(f'preprocessed/I_channel/{image_id}.npy')
print(f"I-channel shape: {i_channel.shape}")  # Should be (256, 256)
print(f"I-channel range: [{i_channel.min():.4f}, {i_channel.max():.4f}]")  # Should be [0, 1]

# Check metadata
with open(f'preprocessed/metadata/{image_id}.json', 'r') as f:
    metadata = json.load(f)
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")