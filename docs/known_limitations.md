# Known Limitations & Honest Scientific Assessment

In accordance with rigorous academic reporting standards, this document records the mathematical and empirical limitations of the completed hybrid watermarking and blind ANN extraction system.

---

## 1. Fundamental Information-Theoretic Limit Under Multi-Copy Averaging Collusion

- Under a scientifically valid Type-I averaging collusion attack, $N$ colluders each hold a copy of the **same host image** embedded with **$N$ distinct, independent watermarks** $W_1, W_2, \dots, W_N$, and average their copies:
  $$\bar{I}(x,y) = I(x,y) + \alpha(x,y)\left(\frac{1}{N}\sum_{k=1}^{N} (M_k(x,y) - 0.5)\right) + \eta(x,y)$$
- Because averaging is completely symmetric with respect to the $N$ colluders:
  1. The victim's watermark amplitude is attenuated by a factor of $1/N$ (power reduced by $1/N^2$).
  2. The remaining $N-1$ independent watermarks act as structured, uncorrelated interference in the exact same spatial/frequency band with combined variance $(N-1)/N^2$.
- **Consequence:** Even with an ideal oracle extractor that knows the clean host $I(x,y)$, the expected Normalized Correlation with any single colluder's watermark is bounded by:
  $$\mathbb{E}[\text{NC}] \le \frac{1}{\sqrt{N + \sigma_\eta^2 / \sigma_w^2}}$$
- On our held-out evaluation:
  - At $N = 2$ colluders, both the Non-Blind extractor and the Blind ANN retain positive correlation ($\text{NC} \approx 0.45–0.55$, high detection rate $\text{CoRI}(2) > 0.90$).
  - At $N = 5$, $\text{NC} \approx 0.25$ (near the statistical detection boundary $\tau = 0.125$).
  - For $N \ge 20$, the victim's signal is diluted below the interference floor ($\text{NC} < 0.10, \text{BER} > 0.40$) for **both** non-blind and blind extractors.
- Achieving asymptotic resistance to large coalitions ($N \ge 20$) requires anti-collusion codes (such as Tardos codes or Boneh–Shaw fingerprinting codes with code length $L = \mathcal{O}(N^2 \log(1/\epsilon))$) rather than uncoded $1,024$-bit spatial tiling.

---

## 2. High-Frequency Attenuation Under Aggressive JPEG Compression & Spatial Blur

- To maintain high visual quality ($\text{PSNR} \approx 41.57\text{ dB}, \text{SSIM} \approx 0.9965$), the adaptive embedder uses a low base amplitude ($\alpha_{\text{base}} = 0.012$, pixel-wise $\alpha(x,y) \in [0.012, 0.036]$), corresponding to $\approx 3–9$ quantization levels out of $255$.
- Because the Catalan + Arnold permutation decorrelates adjacent watermark bits into a white-noise-like spatial pattern, most of the watermark energy resides in high spatial frequencies.
- Consequently:
  - **Clean & Cropping ($10\%, 25\%, 50\%$):** The surviving tiles are unattenuated in high frequencies, so both the Non-Blind extractor and the Blind ANN achieve near-lossless recovery ($\text{NC} > 0.98, \text{CRR} > 0.98$).
  - **Moderate JPEG ($Q = 70$) & Gaussian Noise ($\sigma = 0.05$):** Partial recovery is achieved, with 64-tile aggregation suppressing zero-mean noise.
  - **Aggressive Low-Pass Filtering (Gaussian Blur $k=3$, Median Blur $k=3$, JPEG $Q=50$):** Low-pass filtering directly cancels adjacent opposite-sign pixels within each $32 \times 32$ tile, substantially degrading both Non-Blind and Blind ANN recovery ($\text{NC} \approx 0.15–0.45$).
- **Trade-off:** Increasing $\alpha_{\text{base}}$ (e.g., to $0.03–0.05$) or embedding into $2 \times 2$ / $4 \times 4$ low-frequency sub-blocks improves blur and JPEG $Q=50$ survival at the cost of lowering PSNR from $\sim 41.6\text{ dB}$ to $\sim 35–37\text{ dB}$.

---

## 3. Geometric Desynchronization (Rotation, Scaling, Translation)

- Both the Non-Blind tile folding and the Blind ANN's $8 \times 8$ tile aggregation assume the $256 \times 256$ image grid remains aligned with the $32 \times 32$ tile lattice (which holds for axis-aligned cropping, padding, noise, filtering, JPEG, and collusion).
- Arbitrary rotation, aspect-ratio scaling, or unaligned crop-and-resize without template resynchronization will misalign the $32 \times 32$ periodic grid and require a Spatial Transformer Network (STN) or Fourier–Mellin synchronization pre-stage.

---

## 4. Key Dependence

- The Blind ANN incorporates the inverse permutation $\pi_{\text{key}}^{-1}$ corresponding to the secret parameters (`acm_iterations=10, catalan_iterations=5, catalan_key=7`). Just as with the Non-Blind extractor, the receiver must possess the correct secret key parameters to unscramble the extracted tile into the original binary watermark.
