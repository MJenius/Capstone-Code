# Live Faculty Demonstration Script (`PW26_PAC_01`)

This step-by-step script is designed for a **10–15 minute live capstone evaluation** in front of the faculty panel, demonstrating both the CLI phase runner and the interactive Streamlit workbench.

---

## Part 1 — Launch the Interactive Research Workbench (Recommended Primary Demo)

```bash
streamlit run app.py
```

1. **Tab 1 — Overview:**
   - Point out the live **Blind ANN Status: TRAINED** badge in the sidebar, showing the trained checkpoint epoch, best validation NC, and the $2,960$-sample leakage-free dataset across $800$ unique DIV2K hosts.
2. **Tab 2 — Preprocessing:**
   - Select any held-out test host image (or upload a custom image provided by the panel) and show the $256 \times 256$ bicubic resize, $Y, I, Q$ decomposition, and normalized $I$-channel statistics.
3. **Tab 3 — Watermark & Mosaic:**
   - Show the $32 \times 32$ binary watermark (`PW26_PAC_01`), the 10-iteration Arnold Cat Map output, the 5-iteration keyed Catalan permutation (`key=7`), the $8 \times 8$ mosaic ($64$ redundant copies), and the verified `BER = 0.0000` reversibility check.
4. **Tab 4 — Adaptive Embedding:**
   - Highlight the $7 \times 7$ local variance texture mask $M(x,y)$ and pixel-wise $\alpha(x,y) \in [0.012, 0.036]$, achieving $\text{PSNR} > 41\text{ dB}$ and $\text{SSIM} > 0.996$.
5. **Tab 5 — Attack Simulator:**
   - Select **Cropping** (e.g., $25\%$ or $50\%$ random removal) and show the damaged $I$-channel.
6. **Tab 6 — Non-Blind Extraction:**
   - Explain clearly that this baseline subtracts the **clean original host** (`I_attacked - I_host`) and uses the crop mask, recovering the watermark with $\text{NC} > 0.995$.
7. **Tab 8 — Blind ANN Extraction (Core Contribution Proof):**
   - Switch to **Tab 8**. Emphasize that **only the attacked $256 \times 256$ image** enters `BlindWatermarkExtractor`—no clean host image, no residual, and no crop mask.
   - Show the $32 \times 32$ predicted probability heatmap $P(\text{bit}=1)$, the thresholded $32 \times 32$ binary watermark, and the post-inference comparison against ground truth ($\text{NC} > 0.99$ on clean/cropped test images in $\sim 20\text{ ms}$).
8. **Tab 9 — Performance Comparison:**
   - Inspect the measured comparison table and charts (`nc_comparison.png`, `cropping_curve.png`, `collusion_curve.png`) across all held-out test hosts comparing:
     1. Proposed Hybrid + Blind ANN Extractor
     2. Proposed Hybrid + Non-Blind Extractor
     3. Conventional Center-Tile Baseline
9. **Tab 10 — Full Pipeline Demo:**
   - Toggle between **Blind ANN Extractor** and **Non-Blind Extractor**, select any attack, and click **RUN COMPLETE DEMO** to execute all 6 stages sequentially with live progress and downloadable artifacts.

---

## Part 2 — Stage-by-Stage CLI Verification Commands

Every stage can also be executed independently from the terminal:

```bash
# 1. Environment & dataset audit
python run.py --phase environment

# 2. Host preprocessing & YIQ decomposition
python run.py --phase preprocessing

# 3. Watermark scrambling (Arnold + Catalan + 8x8 Mosaic)
python run.py --phase watermark

# 4. Perceptual luminance-texture adaptive embedding
python run.py --phase embedding

# 5. Standardized attack simulation (crop | jpeg | noise | blur | collusion)
python run.py --phase attacks --attack crop

# 6. Non-blind (host-subtracted) extraction
python run.py --phase extraction

# 7. Generate leakage-free ANN dataset (or --quick for fast subset)
python run.py --phase ann-data

# 8. Train or resume Blind ANN Extractor
python run.py --phase ann-train --quick

# 9. Evaluate Blind ANN on held-out test split
python run.py --phase ann-eval

# 10. Run standalone host-free blind inference on a single image
python run.py --phase ann-inference --image results/phase4_attacks/crop/attacked_i_channel.npy --model models/checkpoints/best_model.pth

# 11. Run 3-way comparison (Blind ANN vs Non-Blind vs Baseline)
python run.py --phase comparison

# 12. Generate 16-panel publication overview & 7 stage figures
python run.py --phase demo

# 13. Run full automated pytest suite
python -m pytest -v
```
