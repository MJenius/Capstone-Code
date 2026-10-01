# Capstone Results & Artifacts Inspection Guide

This guide explains which visual and report artifacts correspond to each phase of the research pipeline.

---

## Directory Overview

All generated outputs are organized under `results/`:

```
results/
├── demo/                                # Primary Capstone presentation artifacts
│   ├── full_pipeline_demo.png           # 12-panel high-res demonstration poster
│   └── demo_report.md                   # End-to-end execution audit report
├── phase0_environment/
│   ├── report.json                      # Machine-readable environment validation
│   └── summary.txt                      # Dataset & watermark readiness summary
├── phase1_preprocessing/
│   ├── phase1_preprocessing_grid.png    # 6-panel YIQ decomposition & normalization
│   ├── host_i_norm.npy                  # Preprocessed normalized host I-channel
│   ├── host_rgb_resized.png             # 256x256 resized host
│   └── report.json                      # Min/Max numerical channel ranges
├── phase2_watermark/
│   ├── phase2_watermark_pipeline.png    # 6-panel dual scrambling & reversibility
│   ├── watermark_binary.npy             # 32x32 binary ground truth
│   ├── watermark_scrambled.npy          # Arnold Cat Map scrambled array
│   ├── watermark_catalan.npy            # Catalan-transformed array
│   ├── watermark_mosaic.npy             # 256x256 8x8 tiled mosaic
│   ├── watermark_mosaic_preview.png     # Visual preview of tiled mosaic
│   └── phase2_report.json               # Reversibility verification metrics (BER=0.0)
├── phase3_embedding/
│   ├── phase3_embedding_pipeline.png    # 6-panel variance mask, alpha & embedding
│   ├── embedded_i_channel.npy           # Embedded 2D I-channel
│   ├── embedded_color_preview.png       # Reconstructed color preview
│   ├── difference_heatmap.png           # 10x amplified embedding alteration heatmap
│   └── report.json                      # Measured PSNR (> 41 dB) & SSIM (> 0.996)
├── phase4_attacks/
│   ├── crop/                            # 25% Random cropping & binary retention mask
│   ├── jpeg/                            # JPEG compression at Q=50
│   ├── noise/                           # Additive Gaussian noise (sigma=0.05)
│   ├── blur/                            # Spatial Gaussian smoothing (k=3)
│   └── collusion/                       # Fingerprinting collusion averaging (N=5)
├── phase5_extraction/
│   ├── phase5_extraction_pipeline.png   # 8-panel extraction & error verification
│   ├── recovered_watermark.npy          # Recovered 32x32 binary watermark
│   ├── recovered_watermark.png          # Visual recovered watermark
│   └── report.json                      # NC, BER, and bit error count
├── phase6_benchmark/
│   ├── benchmark_results.json           # Raw benchmark metrics per image and attack
│   ├── benchmark_results.csv            # Tabular export for spreadsheet analysis
│   ├── summary.md                       # Formatted benchmark summary table
│   └── plots/
│       ├── psnr.png                     # PSNR imperceptibility distribution
│       ├── nc_by_attack.png             # Robustness comparison (Hybrid vs. Baseline)
│       ├── ber_by_attack.png            # Bit error rate across attacks
│       └── collusion_curve.png          # Collusion sensitivity curve (N=2 to 100)
├── phase7_tests/
│   ├── test_report.json                 # Pytest machine-readable test log
│   └── test_report.md                   # Human-readable test pass status
├── phase8_ann_data/
│   ├── dataset_summary.json             # Dataset size and attack distribution
│   ├── metadata.csv                     # Training sample audit trail
│   ├── samples/                         # (input, label) numpy pairs
│   └── visualizations/                  # Verification grids for training pairs
└── phase9_ann/
    └── ann_status.json                  # PLANNED status audit for future ANN
```

---

## Presentation Slides Checklist (PPT Artifact Recommendations)

| PPT Slide Topic | Recommended Graphic File | Supporting Report |
| :--- | :--- | :--- |
| **Complete System Overview** | `results/demo/full_pipeline_demo.png` | `results/demo/demo_report.md` |
| **Color Decomposition** | `results/phase1_preprocessing/phase1_preprocessing_grid.png` | `results/phase1_preprocessing/report.json` |
| **Dual Chaotic Security** | `results/phase2_watermark/phase2_watermark_pipeline.png` | `results/phase2_watermark/phase2_report.json` |
| **Adaptive Embedding** | `results/phase3_embedding/phase3_embedding_pipeline.png` | `results/phase3_embedding/report.json` |
| **Attack Resilience** | `results/phase4_attacks/crop/attack_crop_visualization.png` | `results/phase4_attacks/crop/report.json` |
| **Watermark Extraction** | `results/phase5_extraction/phase5_extraction_pipeline.png` | `results/phase5_extraction/report.json` |
| **Benchmark Robustness** | `results/phase6_benchmark/plots/nc_by_attack.png` | `results/phase6_benchmark/summary.md` |
| **Collusion Sensitivity** | `results/phase6_benchmark/plots/collusion_curve.png` | `results/phase6_benchmark/summary.md` |
