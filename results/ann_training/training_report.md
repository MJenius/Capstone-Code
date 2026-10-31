# ANN Training Report

Generated: 2026-10-31T17:12:10

| Item | Value |
|---|---|
| Device | cpu (torch 2.13.0+cpu) |
| Training samples | 2240 |
| Validation samples | 360 |
| Epochs requested | 10 |
| Epochs completed (total, incl. resumed) | 10 |
| Early stopped | False |
| Quick mode (40 batches/epoch) | False |
| Batch size / LR | 32 / 0.0005 |
| Loss | BCEWithLogits + 0.1 * (1 - softNC) |
| Optimizer / scheduler | AdamW (wd 1e-4) / ReduceLROnPlateau(mode=max on val NC, factor=0.5, patience=1) |
| Selection criterion | max validation NC (test split untouched) |
| Seed | 42 |
| Wall time this run | 1641 s |
| Parameters | 276,466 |

## Best validation epoch (10)

| Metric | Value |
|---|---|
| loss | 0.2538 |
| nc | 0.7759 |
| ber | 0.1088 |
| accuracy | 0.8912 |
| precision | 0.8685 |
| recall | 0.8615 |
| f1 | 0.8606 |

## Per-epoch history

| epoch | train loss | val loss | val NC | val BER | val acc | val F1 | lr | s |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6004 | 0.3871 | 0.6813 | 0.1652 | 0.8348 | 0.8121 | 3.00e-03 | 352.1 |
| 2 | 0.3561 | 0.3307 | 0.7169 | 0.1384 | 0.8616 | 0.8248 | 3.00e-03 | 350.7 |
| 3 | 0.3339 | 0.3138 | 0.7305 | 0.1351 | 0.8649 | 0.8401 | 3.00e-03 | 324.4 |
| 4 | 0.3173 | 0.3034 | 0.7333 | 0.1299 | 0.8701 | 0.8281 | 3.00e-03 | 322.2 |
| 5 | 0.2994 | 0.3008 | 0.7409 | 0.1303 | 0.8697 | 0.8437 | 5.00e-04 | 266.2 |
| 6 | 0.2908 | 0.2831 | 0.7533 | 0.1213 | 0.8787 | 0.8488 | 5.00e-04 | 270.8 |
| 7 | 0.2848 | 0.2814 | 0.7562 | 0.1210 | 0.8790 | 0.8480 | 5.00e-04 | 273.1 |
| 8 | 0.2785 | 0.2679 | 0.7639 | 0.1137 | 0.8863 | 0.8474 | 5.00e-04 | 280.1 |
| 9 | 0.2732 | 0.2638 | 0.7698 | 0.1135 | 0.8865 | 0.8576 | 5.00e-04 | 278.4 |
| 10 | 0.2707 | 0.2538 | 0.7759 | 0.1088 | 0.8912 | 0.8606 | 5.00e-04 | 271.1 |

Note: validation metrics are measured on held-out *validation hosts* and watermark IDs.
They are used for model selection only; final claims use the isolated test split (ann-eval / comparison).
