# Hybrid Image Watermarking Pipeline — Implementation Architecture

This document provides complete Mermaid architecture diagrams representing the completed Capstone Robust Image Watermarking system (`PW26_PAC_01`), including both Non-Blind and Blind ANN extraction paths, leakage-free dataset synthesis, model training, evaluation, and the interactive Streamlit workbench.

---

## 1. Complete Final System Architecture

```mermaid
flowchart TD
    INPUT["INPUT: Host Image (RGB) + Binary Watermark (32x32)"]
    PRE["PREPROCESSING: Resize 256x256 + YIQ Conversion + Normalized I-Channel"]
    WMG["WATERMARK GENERATION: 32x32 Binary Matrix {0,1}"]
    ARN["ARNOLD: Arnold Cat Map Scrambling (10 iterations)"]
    CAT["CATALAN: Keyed Catalan Permutation (5 iterations, key=7)"]
    MOS["MOSAIC: 8x8 Spatial Tiling (256x256, 64 redundant copies)"]
    EMB["ADAPTIVE EMBEDDING: Local 7x7 Variance Mask + Pixel-Wise Alpha (PSNR > 41 dB)"]
    ATK["ATTACK SIMULATION: Cropping / JPEG / Gaussian Noise / Gaussian & Median Blur / Collusion"]
    NB["NON-BLIND EXTRACTION: Host-Subtracted Residual + Mask-Weighted Tile Aggregation + Inverse Catalan & Arnold"]
    ANN["ANN BLIND EXTRACTION: BlindWatermarkExtractor (256x256 Attacked I -> 32x32 Original Watermark, Host-Free)"]
    MET["METRIC ENGINE: PSNR, SSIM, NC, BER, Bit Accuracy, Precision, Recall, F1, CRR, CoRI"]
    CMP["FINAL COMPARISON & INTERACTIVE WORKBENCH (Streamlit app.py + results/final_evaluation/)"]

    INPUT --> PRE
    INPUT --> WMG
    WMG --> ARN
    ARN --> CAT
    CAT --> MOS
    PRE --> EMB
    MOS --> EMB
    EMB --> ATK
    ATK --> NB
    ATK --> ANN
    NB --> MET
    ANN --> MET
    MET --> CMP
```

---

## 2. Detailed End-to-End Component, Training & Evaluation Architecture

Status Legend:
- **IMPLEMENTED & VERIFIED**: Core preprocessing, transforms, adaptive embedding, attack engines, non-blind extractor, and test suite.
- **IMPLEMENTED, TRAINED & EVALUATED**: Leakage-free ANN dataset (`data/ann/`), `BlindWatermarkExtractor` (`models/blind_extractor.py`), trained checkpoints (`models/checkpoints/best_model.pth`), blind inference (`training/inference.py`), 3-way evaluation (`results/final_evaluation/`), and interactive Streamlit workbench (`app.py`).

```mermaid
flowchart TD
    subgraph S1["1. Host & Watermark Preparation [IMPLEMENTED & VERIFIED]"]
        D_RAW["800 Unique DIV2K Hosts\n(Host-Level Split 560 / 120 / 120)"]
        P_YIQ["Resize 256x256 + NTSC YIQ\nNormalized I-Channel [0,1]"]
        W_BIN["32x32 Binary Watermark\n(Disjoint IDs per Split)"]
        W_ACM["Arnold Cat Map (10 iters)"]
        W_CAT["Catalan Permutation (5 iters, k=7)"]
        W_MOS["8x8 Mosaic Grid (256x256, 64 tiles)"]

        D_RAW --> P_YIQ
        W_BIN --> W_ACM --> W_CAT --> W_MOS
    end

    subgraph S2["2. Adaptive Embedding & Attack Suite [IMPLEMENTED & VERIFIED]"]
        E_TEX["7x7 Local Variance Texture Mask M(x,y)"]
        E_EMB["Adaptive Embedder: I' = clip(I + alpha(x,y)*(M - 0.5))"]
        ATK_ALL["Attack Engine: Clean, Crop (10/25/50%),\nJPEG (Q50/Q70), Noise, Gaussian/Median Blur,\nType-I Collusion (N=2..100)"]

        P_YIQ --> E_TEX --> E_EMB
        W_MOS --> E_EMB --> ATK_ALL
    end

    subgraph S3["3. Leakage-Free ANN Dataset & Training [TRAINED & CHECKPOINTED]"]
        DS_GEN["Dataset Generator (training/dataset.py)\n2,960 pairs: 256x256 Attacked I -> 32x32 Watermark"]
        TR_LOOP["PyTorch Trainer (training/trainer.py)\nBCEWithLogits + 0.1*(1 - soft_NC), AdamW, Early Stopping"]
        CKPTS["Saved Checkpoints:\nmodels/checkpoints/best_model.pth\nmodels/checkpoints/last_model.pth"]

        ATK_ALL --> DS_GEN --> TR_LOOP --> CKPTS
    end

    subgraph S4["4. Dual Extraction Paths & Held-Out Evaluation [IMPLEMENTED & EVALUATED]"]
        EX_NB["Non-Blind Extractor (core/extraction.py)\nUses Clean Host I + Crop Mask"]
        EX_BLIND["Blind ANN Extractor (training/inference.py)\nUses ONLY Attacked 256x256 I-Channel"]
        EVAL_3WAY["3-Way Held-Out Test Comparison\n(Blind ANN vs Non-Blind vs Center Baseline)\nMetrics: PSNR, SSIM, NC, BER, Acc, F1, CRR, CoRI"]
        UI_APP["Interactive Streamlit Workbench (app.py)\n10 Stage Tabs + One-Click Complete Demo"]

        ATK_ALL --> EX_NB
        P_YIQ -.->|Host Required| EX_NB
        ATK_ALL --> EX_BLIND
        CKPTS --> EX_BLIND
        EX_NB --> EVAL_3WAY
        EX_BLIND --> EVAL_3WAY
        EVAL_3WAY --> UI_APP
    end

    classDef verified fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef trained fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    class D_RAW,P_YIQ,W_BIN,W_ACM,W_CAT,W_MOS,E_TEX,E_EMB,ATK_ALL,EX_NB verified;
    class DS_GEN,TR_LOOP,CKPTS,EX_BLIND,EVAL_3WAY,UI_APP trained;
```
