# Watermarking System - Phase 4 (Mosaic + Embedding) Implemented

A modular watermarking system implementing Arnold Cat Map (ACM) scrambling for secure watermark embedding in images.

## Features

### Phase 1: Data Preprocessing
- YIQ color space conversion
- Image resizing and normalization
- I-channel extraction
- Dataset splitting (train/val/test)
- Metadata management

### Phase 2: Watermark Scrambling ✓
- Arnold Cat Map (ACM) transformation
- Binary watermark generation
- Perfect reconstruction capability
- Configurable iteration count (encryption key)
- Comprehensive validation suite

### Phase 3: Catalan Transform ✓
- Deterministic Catalan-number-based permutation
- Reversible inverse transform support
- Configurable iteration count and key

### Phase 4: Mosaic + Embedding ✓
- 8x8 tiling of transformed 32x32 watermark to 256x256 mosaic
- Additive embedding into normalized I-channel
- Embedded output saving (`preprocessed/embedded_I_channel/`)
- Color embedded preview generation (`preprocessed/embedded_preview/`)
- Per-image process collage generation (`preprocessed/process_collage/`)
- Per-image embedding metadata (`preprocessed/metadata/embedding_*.json`)

## Project Structure

```
Code/
├── data/
│   ├── watermark/          # Input watermark images
│   ├── scrambled/          # Scrambled watermarks
│   ├── catalan/            # Catalan-transformed watermarks
│   ├── mosaic/             # Generated watermark mosaics
│   └── raw/                # Raw image datasets (not in git)
├── preprocessed/           # Processed data (generated)
│   ├── I_channel/          # Extracted I-channel data
│   ├── embedded_I_channel/ # Embedded I-channel outputs
│   ├── embedded_preview/   # Embedded color PNG previews
│   ├── process_collage/    # Per-image 6-panel process visualizations
│   ├── rgb_256/            # Resized RGB images
│   └── metadata/           # Processing metadata
├── splits/                 # Train/val/test splits
├── utils/                  # Core modules
│   ├── scrambler.py        # Arnold Cat Map implementation
│   ├── processor.py        # Image preprocessing
│   ├── loader.py           # Image loading utilities
│   ├── metadata_mgr.py     # Metadata management
│   └── downloader.py       # Dataset downloader
├── main.py                 # Pipeline orchestration
├── generate_watermark.py   # Watermark generation utility
├── test_phase2.py          # Phase 2 validation
├── test_phase3_mosaic_embedding.py  # Phase 3-4 validation
└── requirements.txt        # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/watermarking-system.git
   cd watermarking-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download datasets** (optional)
   - DIV2K dataset will be automatically downloaded on first run
   - Or manually place images in `data/raw/div2k/` and `data/raw/bossbase/`

## Usage

### Generate Binary Watermark
```bash
python generate_watermark.py
```
- Creates a binary watermark from text "PW26_PAC_01"
- Outputs: `data/watermark/watermark_binary.png` and `.npy`

### Run Phase 2: Watermark Scrambling
```bash
python test_phase2.py
```
- Loads watermark from `data/watermark/`
- Scrambles using Arnold Cat Map (10 iterations)
- Saves to `data/scrambled/`
- Verifies perfect reconstruction

### Run Full Pipeline
```bash
python main.py
```
- Executes Phase 1 (preprocessing) + Phase 2 (scrambling) + Phase 3 (Catalan) + Phase 4 (mosaic + embedding)
- Processes all images in data/raw/
- Generates train/val/test splits

### Run Phase 3 + Phase 4 Validation
```bash
python test_phase3_mosaic_embedding.py
```
- Validates Catalan forward/inverse consistency
- Validates 256x256 mosaic generation from 32x32 watermark
- Validates embedding output shape/range

## Configuration

Edit parameters in the scripts:

**Watermark Settings** (`test_phase2.py`, `main.py`):
- `target_watermark_size`: 32 or 64 (default: 32)
- `acm_iterations`: Number of scrambling iterations (default: 10)

**Watermark Text** (`generate_watermark.py`):
- `text`: Text to embed (default: "PW26_PAC_01")

## Validation

### Test Arnold Cat Map Algorithm
```bash
python utils/scrambler.py
```
- Tests multiple sizes (32×32, 64×64)
- Tests multiple iterations (10, 20, 50)
- Verifies perfect reconstruction

### Expected Output
```
✓ ALL VALIDATION TESTS PASSED
Arnold Cat Map implementation is working correctly!
```

## Algorithm Details

### Arnold Cat Map Transformation
**Forward:**
```
x' = (x + y) mod N
y' = (x + 2y) mod N
```

**Inverse:**
```
x = (2x' - y') mod N
y = (-x' + y') mod N
```

### Properties
- **Chaotic:** Scrambles image appearance
- **Periodic:** Eventually returns to original
- **Bijective:** Perfect one-to-one mapping
- **Deterministic:** Same input + iterations = same output

## File Formats

- **Images:** PNG (visualization), NPY (numerical precision)
- **Metadata:** JSON (human-readable)
- **Splits:** TXT (one ID per line)

## Dependencies

```
opencv-python>=4.8.0    # Image processing
numpy>=1.24.0           # Array operations
scikit-learn>=1.3.0     # Dataset splitting
tqdm>=4.66.0            # Progress bars
Pillow>=10.0.0          # Image generation
```

## Development

### Project Status
- [x] Phase 1: Data Preprocessing
- [x] Phase 2: Watermark Scrambling (Arnold Cat Map)
- [x] Phase 3: Catalan Transform
- [x] Phase 4: Watermark Embedding (with mosaic generation)
- [ ] Phase 5: Watermark Extraction

### Contributing
This is a capstone project. For educational purposes only.

## License

Academic/Educational Use

## Authors

PW26_PAC_01 - Capstone Project 2026

## Acknowledgments

- Arnold Cat Map algorithm
- DIV2K dataset
- BOSSBase dataset (optional)
