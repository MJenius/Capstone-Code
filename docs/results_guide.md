# Capstone Results & Artifacts Inspection Guide

This guide explains where every visual figure, numerical report, model checkpoint, and dataset artifact is stored across the completed pipeline.

---

## Directory Overview

```
Capstone-Code/
├── data/ann/                                    # Leakage-free ANN dataset (800 unique DIV2K hosts)
│   ├── train/inputs.npy, labels.npy             # 2,240 training pairs (560 hosts, 256x256 -> 32x32)
│   ├── val/inputs.npy, labels.npy               # 360 validation pairs (120 hosts, 256x256 -> 32x32)
│   ├── test/inputs.npy, labels.npy              # 360 held-out test pairs (120 hosts, 256x256 -> 32x32)
│   ├── metadata.csv                             # Per-sample host ID, watermark ID, attack type & params
│   ├── dataset_summary.json                     # Verified split counts & zero-leakage validation status
│   └── visualizations/input_label_pairs.png     # Visual grid of (256x256 input, 32x32 label) pairs
├── models/
│   ├── blind_extractor.py                       # BlindWatermarkExtractor PyTorch architecture
│   └── checkpoints/
│       ├── best_model.pth                       # Best validation-NC model weights
│       └── last_model.pth                       # Latest epoch checkpoint (supports --resume)
└── results/
    ├── ann_training/                            # Phase ann-train outputs
    │   ├── training_history.csv                 # Per-epoch loss, NC, BER, accuracy, precision, recall, F1
    │   ├── training_history.json                # JSON training history
    │   ├── loss_curve.png                       # Train vs validation loss curve
    │   ├── validation_accuracy.png              # Train vs validation bit accuracy curve
    │   ├── validation_nc.png                    # Train vs validation NC curve
    │   ├── training_report.md                   # Complete Markdown training summary
    │   └── model_summary.txt                    # Layer-by-layer architecture & parameter count
    ├── ann_evaluation/                          # Phase ann-eval outputs (held-out test split)
    │   ├── ann_test_report.json                 # Held-out test metrics overall & by attack
    │   ├── ann_test_nc_by_attack.png            # Bar chart of held-out test NC by attack type
    │   ├── test_per_attack.csv                  # Per-attack mean & std on test split
    │   └── test_records.csv                     # Individual test sample predictions
    ├── ann_inference/                           # Phase ann-inference outputs (single/batch host-free inference)
    │   ├── input_image.png                      # Attacked 256x256 input fed to the ANN
    │   ├── prediction_probability.png           # 32x32 probability heatmap P(bit=1)
    │   ├── recovered_watermark.png              # Thresholded 32x32 binary watermark
    │   └── inference_report.json                # Inference timing, confidence, and optional post-eval metrics
    ├── final_evaluation/                        # Phase comparison outputs (Blind ANN vs Non-Blind vs Baseline)
    │   ├── comparison_results.csv               # Mean, std, and sample count per method & attack
    │   ├── comparison_results.json              # Full metrics + CRR + CoRI + negative-control FPR
    │   ├── summary.md                           # Formatted academic comparison report
    │   ├── comparison_table.png                 # Visual table of mean NC across attacks
    │   ├── nc_comparison.png                    # Grouped bar chart of NC across all attacks
    │   ├── ber_comparison.png                   # Grouped bar chart of BER across all attacks
    │   ├── psnr_comparison.png                  # Clean PSNR comparison (Hybrid vs Baseline)
    │   ├── ssim_comparison.png                  # Clean SSIM comparison (Hybrid vs Baseline)
    │   ├── cropping_curve.png                   # NC vs cropping area removal (%)
    │   ├── collusion_curve.png                  # NC vs number of colluders N (log scale with 1/sqrt(N) bound)
    │   └── sample_recovery_grid.png             # Visual grid comparing Blind ANN, Non-Blind & Ground Truth
    ├── final_demo/                              # Phase demo outputs (complete 16-panel + 7 stage figures)
    │   ├── full_pipeline.png                    # 16-panel publication overview (both Blind ANN & Non-Blind)
    │   ├── stage_01_preprocessing.png           # Stage 1: RGB, Y, I, Normalized I
    │   ├── stage_02_watermark.png               # Stage 2: Binary, Arnold, Catalan, 8x8 Mosaic
    │   ├── stage_03_embedding.png               # Stage 3: Texture mask, Alpha map, Watermarked RGB, Diff
    │   ├── stage_04_attack.png                  # Stage 4: Clean vs Attacked I-channel & Distortion
    │   ├── stage_05_nonblind_extraction.png     # Stage 5: Non-blind extraction stages
    │   ├── stage_06_ann_blind_extraction.png    # Stage 6: Host-free Blind ANN probability & recovery
    │   ├── stage_07_comparison.png              # Stage 7: Side-by-side Blind ANN vs Non-Blind vs GT
    │   ├── demo_report.md                       # Complete Markdown demonstration report
    │   └── demo_report.json                     # Machine-readable demonstration metrics
    └── final_validation/                        # Automated test & validation suite report
        ├── test_report.json                     # Pytest execution summary
        ├── test_report.md                       # Markdown test report
        └── validation_summary.md                # Final acceptance checklist verification
```
