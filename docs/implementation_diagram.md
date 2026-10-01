# Hybrid Image Watermarking Pipeline — Implementation Architecture

This document provides complete Mermaid architecture diagrams representing the Capstone Robust Image Watermarking system.

---

## 1. System Component & Data Flow Architecture

The diagram below details every component, data transformation, input, output, attack module, evaluation system, validation harness, ANN dataset preparation pipeline, and planned future ANN work.

Status Legend:
- **COMPLETED**: Fully implemented, tested, and validated.
- **PARTIAL / IN PROGRESS**: Functional scaffolding or initial data generation ready.
- **PLANNED / REMAINING**: Future design interfaces (Blind ANN Extractor) clearly separated from current verified code.

```mermaid
graph TD
    %% SUBGRAPH: INPUTS
    subgraph INPUTS["1. Inputs & Ground Truth"]
        D_RAW["DIV2K Dataset\n(800 High-Res PNGs)"]
        W_SRC["Binary Watermark Spec\n(Text: 'PW26_PAC_01')"]
        W_BIN["Binary Watermark Image\n(32x32 uint8 {0, 1})"]
        W_SRC -->|generate_watermark.py| W_BIN
    end

    %% SUBGRAPH: COMPLETED PREPROCESSING
    subgraph COMPLETED_PREPROCESSING["2. Phase 1: Preprocessing [COMPLETED]"]
        P_RESIZE["Bicubic Resizing\n(256x256 BGR)"]
        P_YIQ["Color Conversion\n(BGR to NTSC YIQ)"]
        P_NORM["I-Channel Min-Max Normalization\n([min, max] -> [0.0, 1.0])"]
        P_OUT["Normalized Host I-Channel\n(256x256 float32)"]

        D_RAW --> P_RESIZE
        P_RESIZE --> P_YIQ
        P_YIQ --> P_NORM
        P_NORM --> P_OUT
    end

    %% SUBGRAPH: COMPLETED WATERMARK TRANSFORMS
    subgraph COMPLETED_WATERMARK["3. Phase 2: Watermark Transformation [COMPLETED]"]
        W_ACM["Arnold Cat Map (ACM)\n(Chaotic Scrambling, 10 iters)"]
        W_CAT["Catalan Permutation\n(Blake2b keyed sort, 5 iters, key=7)"]
        W_MOSAIC["8x8 Spatial Mosaic Tiling\n(32x32 -> 256x256, 64 redundant tiles)"]

        W_BIN --> W_ACM
        W_ACM --> W_CAT
        W_CAT --> W_MOSAIC
    end

    %% SUBGRAPH: COMPLETED EMBEDDING
    subgraph COMPLETED_EMBEDDING["4. Phase 3: Adaptive Embedding [COMPLETED]"]
        E_VAR["Local Luminance-Texture Variance\n(BoxFilter 7x7 Window)"]
        E_ALPHA["Adaptive Alpha Field\nalpha_pixel = alpha_base * (1 + sens*mask)"]
        E_FORMULA["Additive Bipolar Embedding\nembedded = host + alpha_pixel * (wm - 0.5)"]
        E_WATERMARKED["Watermarked I-Channel\n(256x256 float32, PSNR > 41 dB)"]

        P_OUT --> E_VAR
        E_VAR --> E_ALPHA
        P_OUT --> E_FORMULA
        W_MOSAIC --> E_FORMULA
        E_ALPHA --> E_FORMULA
        E_FORMULA --> E_WATERMARKED
    end

    %% SUBGRAPH: COMPLETED ATTACKS
    subgraph COMPLETED_ATTACKS["5. Phase 4: Attack Engine Suite [COMPLETED]"]
        direction TB
        ATK_CROP["Cropping Attack\n(Center / Seeded Random / Quadrant)\n(10%, 25%, 50% Area Removal)"]
        ATK_JPEG["JPEG Compression\n(Quality Factor Q=50, 70)"]
        ATK_NOISE["Additive Gaussian Noise\n(zero-mean, sigma=0.05, seeded)"]
        ATK_BLUR["Gaussian / Median Spatial Blur\n(kernel=3x3, sigma=0)"]
        ATK_COLLUSION["Collusion Averaging Attack\n(Standard Fingerprint: 1 victim + N-1 colluders)\n(N in {2, 5, 10, 20, 50, 100})"]

        E_WATERMARKED --> ATK_CROP
        E_WATERMARKED --> ATK_JPEG
        E_WATERMARKED --> ATK_NOISE
        E_WATERMARKED --> ATK_BLUR
        E_WATERMARKED --> ATK_COLLUSION
    end

    %% SUBGRAPH: COMPLETED EXTRACTION
    subgraph COMPLETED_EXTRACTION["6. Phase 5: Non-Blind Extraction [COMPLETED]"]
        EX_SUB["Host-Subtracted Residual Extraction\ndiff = (attacked - host) / alpha_base + 0.5"]
        EX_TILES["Tile Reshape & Weighted Aggregation\n(64 tiles 32x32, weighted by crop mask)"]
        EX_THRESH["Binary Thresholding (> 0.5)"]
        EX_INVCAT["Inverse Catalan Permutation\n(5 iters, key=7)"]
        EX_INVACM["Inverse Arnold Cat Map\n(10 iters)"]
        EX_RECOVERED["Recovered 32x32 Watermark"]

        ATK_CROP --> EX_SUB
        ATK_JPEG --> EX_SUB
        ATK_NOISE --> EX_SUB
        ATK_BLUR --> EX_SUB
        ATK_COLLUSION --> EX_SUB
        P_OUT -.->|Non-Blind Host| EX_SUB

        EX_SUB --> EX_TILES
        EX_TILES --> EX_THRESH
        EX_THRESH --> EX_INVCAT
        EX_INVCAT --> EX_INVACM
        EX_INVACM --> EX_RECOVERED
    end

    %% SUBGRAPH: EVALUATION & VALIDATION
    subgraph COMPLETED_EVAL["7. Phases 6 & 7: Benchmark & Validation [COMPLETED]"]
        EV_METRICS["Evaluation Metrics\n- PSNR & SSIM (Imperceptibility)\n- NC & BER (Robustness)"]
        EV_BENCH["165-Image Test Split Benchmark\n(14 Attack Conditions across DIV2K)"]
        EV_TESTS["Automated Pytest Suite\n(12 Component & Pipeline Tests)"]

        W_BIN -.->|Ground Truth| EV_METRICS
        EX_RECOVERED --> EV_METRICS
        E_WATERMARKED --> EV_METRICS
        P_OUT -.-> EV_METRICS

        EV_METRICS --> EV_BENCH
        EV_TESTS --> EV_BENCH
    end

    %% SUBGRAPH: ANN DATASET PREPARATION
    subgraph COMPLETED_ANN_DATA["8. Phase 8: ANN Dataset Preparation [COMPLETED]"]
        ANN_PAIR["Training Pair Generation\n- Input: Distorted I-Channel (256x256)\n- Label: Clean Recovered Signal (256x256)"]
        ANN_META["Metadata & Audit Trail Logging\n(Sample ID, Seed, Attack Type & Params)"]

        E_WATERMARKED --> ANN_PAIR
        ATK_CROP --> ANN_PAIR
        ATK_JPEG --> ANN_PAIR
        ATK_NOISE --> ANN_PAIR
        ATK_BLUR --> ANN_PAIR
        EX_SUB -->|Clean Signal| ANN_PAIR
        ANN_PAIR --> ANN_META
    end

    %% SUBGRAPH: PLANNED ANN EXTRACTOR
    subgraph PLANNED_ANN["9. Phase 9: Blind ANN Watermark Extractor [PLANNED / REMAINING]"]
        ANN_MODEL["Deep Neural Network Architecture\n(e.g., ResNet/U-Net Encoder-Decoder)\n[INTERFACE READY - WEIGHTS PENDING]"]
        ANN_TRAIN["Supervised Training Harness\n(MSE / BCE Loss + Adam Optimizer)"]
        ANN_BLIND_EVAL["Blind Extraction Robustness Evaluation\n(Host-Free Watermark Extraction)"]

        ANN_PAIR -.->|Training Data| ANN_TRAIN
        ANN_TRAIN -.-> ANN_MODEL
        ANN_MODEL -.-> ANN_BLIND_EVAL
    end

    %% Class styles
    classDef completed fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef partial fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef planned fill:#fbe9e7,stroke:#d84315,stroke-width:2px,stroke-dasharray: 5 5;

    class D_RAW,W_SRC,W_BIN completed;
    class P_RESIZE,P_YIQ,P_NORM,P_OUT completed;
    class W_ACM,W_CAT,W_MOSAIC completed;
    class E_VAR,E_ALPHA,E_FORMULA,E_WATERMARKED completed;
    class ATK_CROP,ATK_JPEG,ATK_NOISE,ATK_BLUR,ATK_COLLUSION completed;
    class EX_SUB,EX_TILES,EX_THRESH,EX_INVCAT,EX_INVACM,EX_RECOVERED completed;
    class EV_METRICS,EV_BENCH,EV_TESTS completed;
    class ANN_PAIR,ANN_META completed;
    class ANN_MODEL,ANN_TRAIN,ANN_BLIND_EVAL planned;
```

---

## 2. Pipeline Execution Sequence

The diagram below maps the execution hierarchy of the master CLI (`run.py`), showing each phase dependency and status.

```mermaid
flowchart TD
    P0["Phase 0: Environment & Dataset Audit<br/><i>(Verify paths, dependencies, watermark availability)</i><br/><b>[COMPLETED]</b>"]
    P1["Phase 1: Image Preprocessing<br/><i>(Resize 256x256, YIQ color conversion, normalize I-channel)</i><br/><b>[COMPLETED]</b>"]
    P2["Phase 2: Watermark Scrambling & Mosaic<br/><i>(Arnold Cat Map, Catalan permutation, 8x8 tiling, roundtrip check)</i><br/><b>[COMPLETED]</b>"]
    P3["Phase 3: Adaptive Embedding<br/><i>(Local variance masking, pixel-wise alpha, imperceptibility audit)</i><br/><b>[COMPLETED]</b>"]
    P4["Phase 4: Attack Suite<br/><i>(Crop, JPEG, Gaussian Noise, Blur, Collusion)</i><br/><b>[COMPLETED]</b>"]
    P5["Phase 5: Non-Blind Extraction<br/><i>(Host-subtracted residual, tile aggregation, inverse transforms)</i><br/><b>[COMPLETED]</b>"]
    P6["Phase 6: Standardized Benchmark<br/><i>(165-image test split, 14 attacks, PSNR/SSIM/NC/BER, plots)</i><br/><b>[COMPLETED]</b>"]
    P7["Phase 7: Automated Test Suite<br/><i>(12 unit/integration tests, verification checks, smoke test)</i><br/><b>[COMPLETED]</b>"]
    P8["Phase 8: ANN Training Dataset Preparation<br/><i>(Distorted input + clean label generation, metadata CSV)</i><br/><b>[COMPLETED]</b>"]
    P9["Phase 9: Blind ANN Watermark Extractor<br/><i>(Deep network model, training loop, blind recovery)</i><br/><b>[PLANNED / REMAINING]</b>"]

    DEMO(["Master CLI Demo Mode<br/><b>python run.py --phase demo</b><br/><i>(Single-command complete research pipeline demonstration)</i>"])

    P0 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
    P5 --> P6
    P6 --> P7
    P7 --> P8
    P8 -.-> P9

    DEMO -.-> P1
    DEMO -.-> P2
    DEMO -.-> P3
    DEMO -.-> P4
    DEMO -.-> P5

    classDef done fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef future fill:#f8d7da,stroke:#dc3545,stroke-width:2px,stroke-dasharray: 4 4;
    classDef highlight fill:#cce5ff,stroke:#004085,stroke-width:3px;

    class P0,P1,P2,P3,P4,P5,P6,P7,P8 done;
    class P9 future;
    class DEMO highlight;
```
