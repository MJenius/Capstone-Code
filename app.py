"""
Interactive Stage-Wise Demonstration & Research Dashboard for the Capstone Project:
"Deep Learning-Based Image Watermarking Using Catalan Transform and Mosaic Generation
for Enhanced Resistance to Cropping and Collusion Attacks"

Launch with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.ann_interface import get_ann_model_info
from core.attacks import AttackRunner
from core.data_sources import BASE_DIR, get_split_hosts, load_host_bgr, make_watermark
from core.embedding import WatermarkEmbedder
from core.evaluation import calculate_psnr, calculate_ssim, evaluate_watermark_recovery
from core.extraction import NonBlindExtractor
from core.preprocessing import (
    bgr_to_yiq,
    denormalize_channel,
    normalize_channel,
    resize_image,
    yiq_to_bgr,
)
from core.watermark import WatermarkTransformer
from training.dataset import ANN_DIR, WatermarkANNDataset
from training.inference import BlindExtractor, image_to_i_channel, score_against_ground_truth
from training.metrics import bit_metrics
from training.trainer import CKPT_DIR, Trainer

st.set_page_config(
    page_title="Capstone Watermarking Research Workbench — PW26_PAC_01",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- helpers
def _arr_to_png_bytes(arr: np.ndarray, is_bgr: bool = False, scale_nearest: Optional[int] = None) -> bytes:
    a = np.asarray(arr)
    if a.ndim == 2:
        if a.dtype != np.uint8:
            a = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
        if scale_nearest:
            a = cv2.resize(a, (scale_nearest, scale_nearest), interpolation=cv2.INTER_NEAREST)
        pil = Image.fromarray(a, mode="L")
    else:
        if is_bgr:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(a)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _heatmap_fig(arr: np.ndarray, title: str, cmap: str = "magma", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _load_selected_host_bgr(uploaded_file, sample_choice: str) -> tuple[np.ndarray, str]:
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is not None:
            return bgr, f"uploaded:{uploaded_file.name}"
    splits = get_split_hosts()
    test_hosts = splits["test"]
    idx = min(int(sample_choice.split("#")[1].split(" ")[0]), len(test_hosts) - 1)
    h = test_hosts[idx]
    return load_host_bgr(h), f"test_host_{h.host_id}"


# --------------------------------------------------------------------------- sidebar
st.sidebar.markdown("## 🛡️ PW26_PAC_01 Workbench")
st.sidebar.caption(
    "**Deep Learning-Based Image Watermarking Using Catalan Transform and Mosaic Generation "
    "for Enhanced Resistance to Cropping and Collusion Attacks**"
)

ann_info = get_ann_model_info()
if ann_info["status"] == "TRAINED":
    st.sidebar.success(
        f"Blind ANN Status: **TRAINED**  \n"
        f"Epoch: `{ann_info.get('epoch')}` | Best Val NC: `{ann_info.get('best_val_nc', 0):.4f}`"
    )
else:
    st.sidebar.warning(f"Blind ANN Status: **{ann_info['status']}**")

st.sidebar.markdown("### Global Parameters")
test_hosts_list = get_split_hosts()["test"]
sample_options = [f"Held-out Test #{i} (Host {h.host_id})" for i, h in enumerate(test_hosts_list[:25])]
selected_sample = st.sidebar.selectbox("Default Held-Out Host Image", sample_options, index=0)
uploaded_host = st.sidebar.file_uploader("Or Upload Custom Host Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

with st.sidebar.expander("Transform & Embedding Hyperparameters", expanded=False):
    acm_iters = st.slider("Arnold Cat Map Iterations", 1, 25, 10)
    cat_iters = st.slider("Catalan Transform Iterations", 1, 15, 5)
    cat_key = st.number_input("Catalan Secret Key", value=7, min_value=1, max_value=999)
    alpha_base = st.slider("Adaptive Alpha Base (α_base)", 0.005, 0.050, 0.012, step=0.001, format="%.3f")
    sensitivity = st.slider("Texture Sensitivity (γ)", 0.5, 4.0, 2.0, step=0.1)
    wm_id_choice = st.selectbox(
        "Watermark Selection",
        ["Canonical ID 0 (PW26_PAC_01)", "Held-out Test Watermark ID 60001", "Held-out Test Watermark ID 60002"],
        index=0,
    )
    wm_id_val = 0 if "ID 0" in wm_id_choice else (60001 if "60001" in wm_id_choice else 60002)

if "exec_logs" not in st.session_state:
    st.session_state["exec_logs"] = []


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state["exec_logs"].append(f"[{ts}] {msg}")


# --------------------------------------------------------------------------- header
st.title("Deep Learning-Based Image Watermarking Using Catalan Transform & Mosaic Generation")
st.markdown(
    "**Academic Capstone Research System (`PW26_PAC_01`)** — Complete Stage-Wise Pipeline, "
    "Perceptual Adaptive Embedding, Standardized Attack Suite, Non-Blind Baseline, and **Host-Free Blind ANN Extractor**."
)

tabs = st.tabs([
    "1. Overview",
    "2. Preprocessing",
    "3. Watermark & Mosaic",
    "4. Adaptive Embedding",
    "5. Attack Simulator",
    "6. Non-Blind Extraction",
    "7. ANN Training",
    "8. Blind ANN Extraction",
    "9. Performance Comparison",
    "10. Full Pipeline Demo",
])

# Shared objects
raw_bgr, host_label = _load_selected_host_bgr(uploaded_host, selected_sample)
resized_bgr = resize_image(raw_bgr, (256, 256))
yiq_img = bgr_to_yiq(resized_bgr)
i_norm, i_min, i_max = normalize_channel(yiq_img[:, :, 1])

wm_gt = make_watermark(wm_id_val)
wm_transformer = WatermarkTransformer(
    watermark_size=32, acm_iterations=acm_iters, catalan_iterations=cat_iters, catalan_key=int(cat_key)
)
wm_fwd = wm_transformer.transform_forward(wm_gt)
embedder = WatermarkEmbedder(alpha_base=alpha_base, sensitivity=sensitivity)
emb_res = embedder.embed_adaptive(i_norm, wm_fwd["watermark_mosaic"])
watermarked_i = emb_res["embedded_i"]

# Reconstruct RGB watermarked image
yiq_wm = yiq_img.copy()
yiq_wm[:, :, 1] = denormalize_channel(watermarked_i, i_min, i_max)
watermarked_bgr = yiq_to_bgr(yiq_wm)

# ===================================================================== TAB 1: OVERVIEW
with tabs[0]:
    st.subheader("Project Objective & Research Problem")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown(
            """
            Digital watermarking systems face a fundamental tension between **imperceptibility** (high PSNR/SSIM)
            and **robustness** against targeted spatial removal (**cropping**) and collaborative averaging (**collusion**).
            Conventional single-region or frequency-localized watermarks are destroyed when their spatial support is cropped.

            ### Proposed Hybrid Methodology
            1. **YIQ Chrominance ($I$-Channel) Isolation:** Targets the $256 \\times 256$ normalized $I$-channel so embedding changes remain perceptually masked.
            2. **Dual Keyed Permutation (Arnold Cat Map + Catalan Transform):** Decorrelates the $32 \\times 32$ binary watermark across all $1,024$ bit positions via a reversible bijection $\\pi_{\\text{key}}$.
            3. **$8 \\times 8$ Spatial Mosaic Redundancy:** Replicates the permuted $32 \\times 32$ tile $64$ times across the $256 \\times 256$ frame so that even $50\\%$ area removal leaves $\\approx 32$ intact tile copies.
            4. **Luminance-Texture Adaptive Embedding:** Modulates pixel-wise strength $\\alpha(x,y) = \\alpha_{\\text{base}}(1 + \\gamma M(x,y))$ using local $7 \\times 7$ variance $M(x,y) \\in [0,1]$, achieving $\\text{PSNR} > 41\\text{ dB}$.
            5. **Dual Extraction Modes:**
               - **Non-Blind Reference Baseline:** Uses the clean original host $I$-channel and crop mask to provide a theoretical upper bound.
               - **Blind ANN Extractor (`BlindWatermarkExtractor`):** A $276\\text{K}$-parameter Mosaic-Aware Multi-Scale Residual CNN that recovers the $32 \\times 32$ binary watermark **directly from the attacked $256 \\times 256$ $I$-channel without the original host image or crop mask**.
            """
        )
    with col_b:
        st.markdown("### Live System Status")
        st.json(ann_info)
        if (ANN_DIR / "dataset_summary.json").exists():
            ds_sum = json.loads((ANN_DIR / "dataset_summary.json").read_text())
            st.metric("ANN Dataset Samples", f"{ds_sum['total_samples']:,}", f"{ds_sum['unique_host_images']} unique DIV2K hosts (0 split leakage)")

    st.markdown("### System Architecture Diagram")
    st.graphviz_chart(
        """
        digraph G {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#f0f4f8", fontname="Helvetica", fontsize=10];
            H [label="Host RGB\\n(Resized 256x256)"];
            YIQ [label="YIQ Conversion\\nNormalized I-Channel"];
            W [label="Binary Watermark\\n(32x32)"];
            ACM [label="Arnold Cat Map\\n(10 iters)"];
            CAT [label="Catalan Permutation\\n(5 iters, key=7)"];
            MOS [label="8x8 Mosaic Grid\\n(256x256, 64 tiles)"];
            EMB [label="Texture-Adaptive Embedder\\nI' = clip(I + α(x,y)(M-0.5))", fillcolor="#d9edf7"];
            ATK [label="Attack Suite\\nCrop / JPEG / Noise / Blur / Collusion", fillcolor="#fcf8e3"];
            NB [label="Non-Blind Extractor\\n(Uses Clean Host + Mask)", fillcolor="#e8eaf6"];
            ANN [label="Blind ANN Extractor\\n(Host-Free 256x256 -> 32x32)", fillcolor="#dff0d8"];
            MET [label="Evaluation Engine\\nPSNR, SSIM, NC, BER, CRR, CoRI"];
            H -> YIQ -> EMB;
            W -> ACM -> CAT -> MOS -> EMB;
            EMB -> ATK;
            ATK -> NB -> MET;
            ATK -> ANN -> MET;
        }
        """
    )

# ===================================================================== TAB 2: PREPROCESSING
with tabs[1]:
    st.subheader(f"Stage 1 — Host Image Validation, Resizing & YIQ Decomposition (`{host_label}`)")
    if st.button("▶ Run Preprocessing Stage", key="btn_preproc"):
        _log(f"Executed preprocessing on {host_label}: raw shape {raw_bgr.shape} -> (256, 256)")
    c1, c2, c3 = st.columns(3)
    c1.image(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), caption=f"Original Host Image {raw_bgr.shape}", use_container_width=True)
    c2.image(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB), caption="Bicubic Resized RGB (256×256)", use_container_width=True)
    c3.image((i_norm * 255).astype(np.uint8), caption="Normalized I-Channel [0, 1]", use_container_width=True)

    c4, c5, c6 = st.columns(3)
    c4.pyplot(_heatmap_fig(yiq_img[:, :, 0], "Y (Luminance) Channel", cmap="gray"))
    c5.pyplot(_heatmap_fig(yiq_img[:, :, 1], f"I (In-phase Chrominance) [{i_min:.1f}, {i_max:.1f}]", cmap="coolwarm"))
    c6.pyplot(_heatmap_fig(yiq_img[:, :, 2], "Q (Quadrature Chrominance)", cmap="PiYG"))

    st.markdown("#### Channel Statistics")
    st.dataframe(
        pd.DataFrame([
            {"Channel": "Y (Luminance)", "Min": float(yiq_img[:, :, 0].min()), "Max": float(yiq_img[:, :, 0].max()), "Mean": float(yiq_img[:, :, 0].mean()), "Std": float(yiq_img[:, :, 0].std())},
            {"Channel": "I (Raw Chrominance)", "Min": i_min, "Max": i_max, "Mean": float(yiq_img[:, :, 1].mean()), "Std": float(yiq_img[:, :, 1].std())},
            {"Channel": "Q (Raw Chrominance)", "Min": float(yiq_img[:, :, 2].min()), "Max": float(yiq_img[:, :, 2].max()), "Mean": float(yiq_img[:, :, 2].mean()), "Std": float(yiq_img[:, :, 2].std())},
            {"Channel": "Normalized I [0,1]", "Min": float(i_norm.min()), "Max": float(i_norm.max()), "Mean": float(i_norm.mean()), "Std": float(i_norm.std())},
        ]),
        use_container_width=True,
    )

# ===================================================================== TAB 3: WATERMARK & MOSAIC
with tabs[2]:
    st.subheader("Stage 2 — Binary Watermark Scrambling (Arnold Cat Map + Catalan Transform) & 8×8 Mosaic")
    rev = wm_transformer.verify_reversibility(wm_gt)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Watermark Shape", "32 × 32 (1024 bits)")
    m2.metric("Arnold / Catalan Params", f"ACM={acm_iters}, Cat={cat_iters}, Key={cat_key}")
    m3.metric("Mosaic Dimensions", "256 × 256 (8×8 = 64 tiles)")
    m4.metric("Bit-Exact Reversibility", "PASS (BER=0.0000)" if rev["exact_reversibility"] else "FAIL")

    w1, w2, w3, w4 = st.columns(4)
    w1.image(_arr_to_png_bytes(rev["watermark_binary"], scale_nearest=256), caption="1. Original Binary Watermark (32×32)", use_container_width=True)
    w2.image(_arr_to_png_bytes(rev["watermark_scrambled"], scale_nearest=256), caption=f"2. Arnold Scrambled ({acm_iters} iters)", use_container_width=True)
    w3.image(_arr_to_png_bytes(rev["watermark_catalan"], scale_nearest=256), caption=f"3. Catalan Permuted ({cat_iters} iters, key={cat_key})", use_container_width=True)
    w4.image(_arr_to_png_bytes(rev["watermark_mosaic"]), caption="4. 8×8 Spatial Mosaic (256×256, 64 copies)", use_container_width=True)

# ===================================================================== TAB 4: ADAPTIVE EMBEDDING
with tabs[3]:
    st.subheader("Stage 3 — Perceptual Luminance-Texture Adaptive Embedding")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Clean PSNR (I-channel)", f"{emb_res['psnr']:.2f} dB")
    e2.metric("Clean SSIM (I-channel)", f"{emb_res['ssim']:.4f}")
    e3.metric("Adaptive α Range", f"[{emb_res['alpha_min']:.4f}, {emb_res['alpha_max']:.4f}]")
    e4.metric("Mean |I' - I|", f"{emb_res['diff_mean']:.5f}")

    p1, p2, p3 = st.columns(3)
    p1.image(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB), caption="Clean Host RGB (256×256)", use_container_width=True)
    p2.image(cv2.cvtColor(watermarked_bgr, cv2.COLOR_BGR2RGB), caption=f"Watermarked RGB (PSNR={emb_res['psnr']:.2f} dB)", use_container_width=True)
    p3.image(_arr_to_png_bytes(wm_fwd["watermark_mosaic"]), caption="Embedded 8×8 Watermark Mosaic", use_container_width=True)

    p4, p5, p6 = st.columns(3)
    p4.pyplot(_heatmap_fig(emb_res["texture_mask"], "Local 7×7 Variance Texture Mask M(x,y)", cmap="viridis"))
    p5.pyplot(_heatmap_fig(emb_res["alpha_map"], "Pixel-Wise Adaptive Strength α(x,y)", cmap="magma"))
    p6.pyplot(_heatmap_fig(emb_res["diff_map"], "Absolute Embedding Difference |I' - I|", cmap="inferno"))

    st.download_button(
        "⬇ Download Watermarked RGB Image (PNG)",
        data=_arr_to_png_bytes(watermarked_bgr, is_bgr=True),
        file_name=f"watermarked_{host_label}.png",
        mime="image/png",
    )

# ===================================================================== TAB 5: ATTACK SIMULATOR
with tabs[4]:
    st.subheader("Stage 4 — Standardized Geometric, Signal & Collaborative Attack Simulator")
    atk_col1, atk_col2 = st.columns([1, 2])
    with atk_col1:
        atk_type = st.selectbox(
            "Select Attack Type",
            ["Cropping", "JPEG Compression", "Gaussian Noise", "Gaussian Blur", "Median Blur", "Collusion (Multi-Copy Averaging)"],
        )
        atk_runner = AttackRunner(target_size=256)
        if atk_type == "Cropping":
            crop_mode = st.selectbox("Crop Mode", ["random", "center", "quadrant"])
            crop_int = st.slider("Area Removal Fraction", 0.05, 0.75, 0.25, step=0.05)
            crop_fill = st.selectbox("Fill Removed Region With", ["zero", "noise"])
            crop_seed = st.number_input("Random Seed", value=42, step=1)
            atk_out = atk_runner.run_crop(watermarked_i, mode=crop_mode, intensity=crop_int, fill_val=crop_fill, seed=int(crop_seed))
            atk_desc = f"Crop ({crop_mode}, {crop_int*100:.0f}% removed, fill={crop_fill})"
        elif atk_type == "JPEG Compression":
            jpeg_q = st.slider("JPEG Quality Factor (Q)", 10, 95, 70, step=5)
            atk_out = atk_runner.run_jpeg(watermarked_i, quality=int(jpeg_q))
            atk_desc = f"JPEG Compression (Q={jpeg_q})"
        elif atk_type == "Gaussian Noise":
            noise_sig = st.slider("Gaussian Noise Sigma (σ)", 0.005, 0.10, 0.03, step=0.005, format="%.3f")
            atk_out = atk_runner.run_noise(watermarked_i, sigma=float(noise_sig), seed=42)
            atk_desc = f"Gaussian Noise (σ={noise_sig:.3f})"
        elif atk_type == "Gaussian Blur":
            blur_k = st.selectbox("Gaussian Kernel Size", [3, 5, 7], index=0)
            atk_out = atk_runner.run_blur(watermarked_i, kernel_size=int(blur_k))
            atk_desc = f"Gaussian Blur (k={blur_k}×{blur_k})"
        elif atk_type == "Median Blur":
            med_k = st.selectbox("Median Filter Kernel Size", [3, 5], index=0)
            attacked_med = atk_runner.signaller.apply_median_blur(watermarked_i, kernel_size=int(med_k))
            atk_out = {"attacked_image": attacked_med, "mask": None, "attack_type": "median_blur", "kernel_size": med_k,
                       "mean_distortion": float(np.mean(np.abs(attacked_med - watermarked_i)))}
            atk_desc = f"Median Blur (k={med_k}×{med_k})"
        else:
            n_col = st.select_slider("Number of Colluders (N)", options=[2, 5, 10, 20, 50, 100], value=5)
            col_noise = st.slider("Collaborative Noise Std", 0.0, 0.03, 0.01, step=0.005)
            from core.hybrid_system import HybridSystem
            hs = HybridSystem(alpha_base=alpha_base, sensitivity=sensitivity, acm_iterations=acm_iters,
                              catalan_iterations=cat_iters, catalan_key=int(cat_key))
            copies_info = hs.collusion_copies(i_norm, wm_gt, int(n_col), seed=42)
            colluded_img = atk_runner.colluder.simulate_collusion(copies_info["copies"], noise_std=float(col_noise), seed=42)
            atk_out = {"attacked_image": colluded_img, "mask": None, "attack_type": "collusion",
                       "n_colluders": int(n_col), "colluder_watermark_ids": copies_info["colluder_wm_ids"][:5],
                       "mean_distortion": float(np.mean(np.abs(colluded_img - watermarked_i)))}
            atk_desc = f"Type-I Averaging Collusion (N={n_col} independent watermarks)"

    attacked_i_sim = atk_out["attacked_image"]
    attacked_mask_sim = atk_out.get("mask")
    with atk_col2:
        ac1, ac2, ac3 = st.columns(3)
        ac1.image(_arr_to_png_bytes(watermarked_i), caption="Watermarked I-Channel (Before Attack)", use_container_width=True)
        ac2.image(_arr_to_png_bytes(attacked_i_sim), caption=f"Attacked I-Channel: {atk_desc}", use_container_width=True)
        ac3.pyplot(_heatmap_fig(np.abs(attacked_i_sim - watermarked_i), "|Attacked - Watermarked|", cmap="inferno"))
        st.json({k: v for k, v in atk_out.items() if k not in ("attacked_image", "mask")})
        st.download_button(
            "⬇ Download Attacked I-Channel (PNG)",
            data=_arr_to_png_bytes(attacked_i_sim),
            file_name=f"attacked_{atk_out['attack_type']}.png",
            mime="image/png",
        )

# ===================================================================== TAB 6: NON-BLIND EXTRACTION
with tabs[5]:
    st.subheader("Stage 5 — NON-BLIND (Host-Subtracted) Watermark Extraction")
    st.info(
        "**NON-BLIND MODE:** This baseline extractor requires the **original unwatermarked host I-channel** "
        "`I_host` (and the crop mask when cropped) to compute the residual `(I_attacked - I_host) / α`. "
        "It serves as an upper-bound reference for the blind ANN extractor."
    )
    nb_mode = st.radio(
        "Alpha Compensation Mode",
        ["Pixel-wise α(x,y) compensation (Exact inverse of adaptive embedder)", "Scalar α_base division (Legacy baseline)"],
        horizontal=True,
    )
    from core.hybrid_system import HybridSystem
    hs_nb = HybridSystem(alpha_base=alpha_base, sensitivity=sensitivity, acm_iterations=acm_iters,
                         catalan_iterations=cat_iters, catalan_key=int(cat_key))
    nb_alpha_mode = "pixel" if "Pixel-wise" in nb_mode else "base"
    nb_detail = hs_nb.extract_nonblind(attacked_i_sim, i_norm, mask=attacked_mask_sim, alpha_mode=nb_alpha_mode)
    nb_inv = wm_transformer.transform_inverse(nb_detail["tile_bin"])
    nb_metrics = bit_metrics(nb_inv["recovered_binary"], wm_gt)

    nb1, nb2, nb3, nb4 = st.columns(4)
    nb1.metric("Normalized Correlation (NC)", f"{nb_metrics['nc']:.4f}")
    nb2.metric("Bit Error Rate (BER)", f"{nb_metrics['ber']:.4f}")
    nb3.metric("Bit Accuracy", f"{nb_metrics['accuracy']*100:.2f}%")
    nb4.metric("F1 Score", f"{nb_metrics['f1']:.4f}")

    g1, g2, g3, g4, g5, g6 = st.columns(6)
    g1.image(_arr_to_png_bytes(attacked_i_sim), caption="1. Attacked I-Channel", use_container_width=True)
    g2.image(_arr_to_png_bytes(np.clip(nb_detail["residual"], 0, 1)), caption="2. Host-Subtracted Residual", use_container_width=True)
    g3.image(_arr_to_png_bytes(np.clip(nb_detail["tile_soft"], 0, 1), scale_nearest=256), caption="3. Aggregated 32×32 Tile", use_container_width=True)
    g4.image(_arr_to_png_bytes(nb_inv["recovered_scrambled"], scale_nearest=256), caption="4. Inverse Catalan", use_container_width=True)
    g5.image(_arr_to_png_bytes(nb_inv["recovered_binary"], scale_nearest=256), caption="5. Inverse Arnold (Final)", use_container_width=True)
    g6.image(_arr_to_png_bytes(wm_gt, scale_nearest=256), caption="6. Ground-Truth Watermark", use_container_width=True)

# ===================================================================== TAB 7: ANN TRAINING
with tabs[6]:
    st.subheader("Stage 6 — Blind ANN Dataset Overview, Architecture & Live Training")
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.markdown("#### Dataset & Split Summary (`data/ann/dataset_summary.json`)")
        if (ANN_DIR / "dataset_summary.json").exists():
            ds = json.loads((ANN_DIR / "dataset_summary.json").read_text())
            st.write({
                "total_samples": ds["total_samples"],
                "unique_host_images": ds["unique_host_images"],
                "unique_watermark_ids": ds["unique_watermark_ids"],
                "samples_per_split": ds["samples_per_split"],
                "hosts_per_split": ds["hosts_per_split"],
                "host_overlap_between_splits": ds["host_overlap_between_splits"],
                "image_dimensions": ds["image_dimensions"],
                "label_dimensions": ds["label_dimensions"],
                "data_validation_status": ds["data_validation_status"],
            })
            if (ANN_DIR / "visualizations" / "input_label_pairs.png").exists():
                st.image(str(ANN_DIR / "visualizations" / "input_label_pairs.png"), caption="Sample (256×256 Attacked Input → 32×32 Binary Watermark Label) Pairs")
        else:
            st.warning("Dataset not found. Click below to generate `data/ann/`.")
            if st.button("Generate ANN Dataset Now"):
                from pipeline.ann_phases import phase_ann_data
                with st.spinner("Generating ANN dataset..."):
                    phase_ann_data(quick=True)
                st.rerun()

    with col_t2:
        st.markdown("#### Model Architecture (`BlindWatermarkExtractor`)")
        from models.blind_extractor import build_model, count_parameters
        _m = build_model()
        st.code(f"{_m}\n\nTotal Trainable Parameters: {count_parameters(_m):,}", language="text")

    st.markdown("---")
    st.markdown("#### Run or Resume PyTorch Training from Dashboard")
    tc1, tc2, tc3, tc4 = st.columns(4)
    ui_epochs = tc1.number_input("Epochs", min_value=1, max_value=50, value=2)
    ui_batch = tc2.selectbox("Batch Size", [16, 32], index=1)
    ui_lr = tc3.number_input("Learning Rate", value=0.002, format="%.4f")
    ui_quick = tc4.checkbox("Quick Mode (25 batches/epoch for live demo)", value=True)

    if st.button("🚀 Start / Fine-Tune ANN Training"):
        prog_bar = st.progress(0.0, text="Initializing PyTorch Trainer...")
        status_box = st.empty()
        tr_ds = WatermarkANNDataset("train", augment=True)
        va_ds = WatermarkANNDataset("val")
        trainer = Trainer(
            tr_ds, va_ds, epochs=int(ui_epochs), batch_size=int(ui_batch), lr=float(ui_lr),
            results_dir=BASE_DIR / "results" / "ann_training",
            max_train_batches=25 if ui_quick else None,
        )
        if (CKPT_DIR / "last_model.pth").exists():
            trainer.resume(CKPT_DIR / "last_model.pth")
            trainer.cfg["epochs"] = trainer.start_epoch + int(ui_epochs)

        total_ep = trainer.cfg["epochs"] - trainer.start_epoch
        done_ep = [0]

        def _on_ep(rec):
            done_ep[0] += 1
            prog_bar.progress(min(done_ep[0] / max(total_ep, 1), 1.0),
                              text=f"Epoch {rec['epoch']}: Val NC={rec['val_nc']:.4f}, Val BER={rec['val_ber']:.4f}")
            status_box.write(rec)

        res_fit = trainer.fit(on_epoch=_on_ep)
        trainer.write_history()
        _log(f"Trained BlindWatermarkExtractor for {done_ep[0]} epoch(s): best val NC={res_fit['best_val_nc']:.4f}")
        st.success(f"Training finished! Best Validation NC: {res_fit['best_val_nc']:.4f}")

    hist_json = BASE_DIR / "results" / "ann_training" / "training_history.json"
    if hist_json.exists():
        st.markdown("#### Recorded Training History & Curves (`results/ann_training/`)")
        hist_df = pd.DataFrame(json.loads(hist_json.read_text()))
        st.dataframe(hist_df, use_container_width=True)
        hc1, hc2, hc3 = st.columns(3)
        for col_widget, png_name, cap in [
            (hc1, "loss_curve.png", "Training & Validation Loss"),
            (hc2, "validation_nc.png", "Validation Normalized Correlation (NC)"),
            (hc3, "validation_accuracy.png", "Validation Bit Accuracy"),
        ]:
            p = BASE_DIR / "results" / "ann_training" / png_name
            if p.exists():
                col_widget.image(str(p), caption=cap, use_container_width=True)

# ===================================================================== TAB 8: BLIND ANN EXTRACTION
with tabs[7]:
    st.subheader("Stage 7 — BLIND ANN Watermark Extraction (Host-Free Inference)")
    st.success(
        "**BLIND INFERENCE GUARANTEE:** The neural network receives **ONLY** the $256 \\times 256$ attacked "
        "image channel. It does **NOT** receive the clean host image, embedding residual, or crop mask. "
        "Ground-truth watermark comparison is performed strictly *after* inference."
    )
    ckpt_files = sorted([p for p in CKPT_DIR.glob("*.pth") if "backup" not in p.name])
    if not ckpt_files:
        st.error("No trained checkpoint found in `models/checkpoints/`. Run training in Tab 7 or via `python run.py --phase ann-train`.")
    else:
        ckpt_labels = [p.relative_to(BASE_DIR).as_posix() for p in ckpt_files]
        ckpt_sel_rel = st.selectbox("Select Model Checkpoint", ckpt_labels, index=0)
        ckpt_sel = str(BASE_DIR / ckpt_sel_rel)
        input_source = st.radio(
            "Blind Inference Input Source",
            ["Use current attacked image from Tab 5 (Attack Simulator)", "Upload an external watermarked/attacked image"],
            horizontal=True,
        )
        ext_upload = None
        if "Upload" in input_source:
            ext_upload = st.file_uploader("Upload Attacked Image (PNG/JPG/NPY)", type=["png", "jpg", "jpeg", "npy"], key="blind_up")

        blind_ext = BlindExtractor(ckpt_sel)
        if ext_upload is not None:
            if ext_upload.name.endswith(".npy"):
                blind_inp = np.load(ext_upload)
            else:
                fb = np.asarray(bytearray(ext_upload.read()), dtype=np.uint8)
                blind_inp = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        else:
            blind_inp = attacked_i_sim

        pred_out = blind_ext.predict(blind_inp)
        post_metrics = score_against_ground_truth(pred_out["binary"], wm_gt)
        err_map_blind = np.abs(pred_out["binary"].astype(np.float32) - wm_gt.astype(np.float32))

        bm1, bm2, bm3, bm4, bm5, bm6, bm7 = st.columns(7)
        bm1.metric("Blind NC", f"{post_metrics['nc']:.4f}")
        bm2.metric("Blind BER", f"{post_metrics['ber']:.4f}")
        bm3.metric("Bit Accuracy", f"{post_metrics['accuracy']*100:.2f}%")
        bm4.metric("Precision", f"{post_metrics['precision']:.4f}")
        bm5.metric("Recall", f"{post_metrics['recall']:.4f}")
        bm6.metric("F1 Score", f"{post_metrics['f1']:.4f}")
        bm7.metric("Inference Time", f"{pred_out['inference_seconds']*1000:.1f} ms")

        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
        bc1.image(_arr_to_png_bytes(pred_out["input_i_channel"]), caption="1. Sole Input to ANN (256×256)", use_container_width=True)
        bc2.pyplot(_heatmap_fig(pred_out["probability"], "2. ANN Probability Map P(bit=1)", cmap="magma", vmin=0, vmax=1))
        bc3.image(_arr_to_png_bytes(pred_out["binary"], scale_nearest=256), caption="3. Recovered Binary Watermark (32×32)", use_container_width=True)
        bc4.image(_arr_to_png_bytes(wm_gt, scale_nearest=256), caption="4. Ground-Truth Watermark (Post-Eval)", use_container_width=True)
        bc5.pyplot(_heatmap_fig(err_map_blind, f"5. Bit Discrepancy ({int(err_map_blind.sum())}/1024 errors)", cmap="magma", vmin=0, vmax=1))

        st.download_button(
            "⬇ Download Blind-Recovered Watermark (PNG)",
            data=_arr_to_png_bytes(pred_out["binary"], scale_nearest=256),
            file_name="blind_recovered_watermark.png",
            mime="image/png",
        )

# ===================================================================== TAB 9: PERFORMANCE COMPARISON
with tabs[8]:
    st.subheader("Stage 8 — Held-Out Test Split Evaluation: Blind ANN vs. Non-Blind vs. Center-Tile Baseline")
    comp_csv = BASE_DIR / "results" / "final_evaluation" / "comparison_results.csv"
    comp_json = BASE_DIR / "results" / "final_evaluation" / "comparison_results.json"
    if comp_csv.exists() and comp_json.exists():
        df_comp = pd.read_csv(comp_csv)
        meta_comp = json.loads(comp_json.read_text())
        st.caption(
            f"Measured on **{meta_comp['n_test_hosts']} held-out test hosts** (zero host or watermark leakage). "
            "Note: Non-blind methods have access to the original clean host image; the Blind ANN does not."
        )
        sel_attacks = st.multiselect(
            "Filter Attack Conditions",
            options=list(df_comp["attack"].unique()),
            default=list(df_comp["attack"].unique()),
        )
        df_view = df_comp[df_comp["attack"].isin(sel_attacks)]
        st.dataframe(df_view, use_container_width=True)

        st.markdown("#### Research-Specific Metrics: Cropping Resistance Ratio (CRR) & Collusion Robustness Index (CoRI)")
        st.json(meta_comp["research_metrics"])

        pc1, pc2 = st.columns(2)
        for widget, fn, title in [
            (pc1, "nc_comparison.png", "Normalized Correlation (NC) Across Attacks"),
            (pc2, "ber_comparison.png", "Bit Error Rate (BER) Across Attacks"),
            (pc1, "cropping_curve.png", "Cropping Robustness Curve"),
            (pc2, "collusion_curve.png", "Collusion Robustness Curve vs 1/√N Bound"),
        ]:
            fp = BASE_DIR / "results" / "final_evaluation" / fn
            if fp.exists():
                widget.image(str(fp), caption=title, use_container_width=True)

        grid_fp = BASE_DIR / "results" / "final_evaluation" / "sample_recovery_grid.png"
        if grid_fp.exists():
            st.image(str(grid_fp), caption="Held-Out Test Sample Recoveries (Blind ANN vs Non-Blind vs Ground Truth)", use_container_width=True)
    else:
        st.info("Comparison results not yet generated. Click below or run `python run.py --phase comparison`.")
        if st.button("Run Held-Out Comparison Now (Quick 10-Host Subset)"):
            from pipeline.ann_phases import phase_comparison
            with st.spinner("Evaluating Blind ANN, Non-Blind, and Baseline on held-out test hosts..."):
                phase_comparison(quick=True)
            st.rerun()

# ===================================================================== TAB 10: FINAL PIPELINE DEMO
with tabs[9]:
    st.subheader("Stage 9 — One-Click End-to-End Pipeline Demonstration")
    dcol1, dcol2, dcol3 = st.columns(3)
    demo_extractor_mode = dcol1.radio(
        "Select Extractor Mode for Complete Demo",
        ["Blind ANN Extractor (Host-Free PyTorch Model)", "Non-Blind Extractor (Uses Clean Host Image)"],
    )
    demo_attack_choice = dcol2.selectbox(
        "Select Attack Condition for Complete Demo",
        ["25% Random Cropping", "50% Random Cropping", "10% Random Cropping", "No Attack (Clean)",
         "JPEG Compression (Q=70)", "JPEG Compression (Q=50)", "Gaussian Noise (σ=0.03)",
         "Gaussian Blur (k=3)", "Median Blur (k=3)", "Collusion Averaging (N=2)", "Collusion Averaging (N=5)"],
        index=0,
    )
    run_complete_clicked = dcol3.button("▶ RUN COMPLETE DEMO", type="primary", use_container_width=True)

    if run_complete_clicked or "last_complete_demo" not in st.session_state:
        prog = st.progress(0.0, text="Stage 1/6: Preprocessing host image...")
        t_start = time.perf_counter()

        # 1. Preprocessing
        time.sleep(0.05)
        prog.progress(0.18, text="Stage 2/6: Arnold + Catalan scrambling & 8×8 mosaic generation...")
        # 2. Watermark transform
        fwd_demo = wm_transformer.transform_forward(wm_gt)
        prog.progress(0.36, text="Stage 3/6: Perceptual luminance-texture adaptive embedding...")
        # 3. Embedding
        emb_demo = embedder.embed_adaptive(i_norm, fwd_demo["watermark_mosaic"])
        wm_i_demo = emb_demo["embedded_i"]
        prog.progress(0.55, text=f"Stage 4/6: Applying {demo_attack_choice}...")

        # 4. Attack
        ar = AttackRunner(target_size=256)
        if "25%" in demo_attack_choice:
            ar_res = ar.run_crop(wm_i_demo, mode="random", intensity=0.25, seed=42)
        elif "50%" in demo_attack_choice:
            ar_res = ar.run_crop(wm_i_demo, mode="random", intensity=0.50, seed=42)
        elif "10%" in demo_attack_choice:
            ar_res = ar.run_crop(wm_i_demo, mode="random", intensity=0.10, seed=42)
        elif "Clean" in demo_attack_choice:
            ar_res = {"attacked_image": wm_i_demo.copy(), "mask": None, "attack_type": "clean"}
        elif "Q=70" in demo_attack_choice:
            ar_res = ar.run_jpeg(wm_i_demo, quality=70)
        elif "Q=50" in demo_attack_choice:
            ar_res = ar.run_jpeg(wm_i_demo, quality=50)
        elif "Noise" in demo_attack_choice:
            ar_res = ar.run_noise(wm_i_demo, sigma=0.03, seed=42)
        elif "Gaussian Blur" in demo_attack_choice:
            ar_res = ar.run_blur(wm_i_demo, kernel_size=3)
        elif "Median" in demo_attack_choice:
            ar_res = {"attacked_image": ar.signaller.apply_median_blur(wm_i_demo, 3), "mask": None, "attack_type": "median"}
        elif "N=2" in demo_attack_choice:
            from core.hybrid_system import HybridSystem
            hs_d = HybridSystem(alpha_base=alpha_base, sensitivity=sensitivity, acm_iterations=acm_iters,
                                catalan_iterations=cat_iters, catalan_key=int(cat_key))
            cp = hs_d.collusion_copies(i_norm, wm_gt, 2, seed=42)
            ar_res = {"attacked_image": ar.colluder.simulate_collusion(cp["copies"], noise_std=0.01, seed=42),
                      "mask": None, "attack_type": "collusion_2"}
        else:
            from core.hybrid_system import HybridSystem
            hs_d = HybridSystem(alpha_base=alpha_base, sensitivity=sensitivity, acm_iterations=acm_iters,
                                catalan_iterations=cat_iters, catalan_key=int(cat_key))
            cp = hs_d.collusion_copies(i_norm, wm_gt, 5, seed=42)
            ar_res = {"attacked_image": ar.colluder.simulate_collusion(cp["copies"], noise_std=0.01, seed=42),
                      "mask": None, "attack_type": "collusion_5"}

        atk_img_demo = ar_res["attacked_image"]
        atk_mask_demo = ar_res.get("mask")

        prog.progress(0.80, text="Stage 5/6: Running watermark extraction...")
        # Run BOTH so user can see active choice and compare
        nb_ext_obj = NonBlindExtractor(alpha_base=alpha_base, watermark_size=32, acm_iterations=acm_iters,
                                       catalan_iterations=cat_iters, catalan_key=int(cat_key))
        nb_out_demo = nb_ext_obj.extract(atk_img_demo, i_norm, mask=atk_mask_demo, ground_truth_binary=wm_gt)
        nb_m_demo = bit_metrics(nb_out_demo["recovered_binary"], wm_gt)

        blind_prob_demo = None
        blind_bin_demo = None
        blind_m_demo = None
        if (CKPT_DIR / "best_model.pth").exists():
            b_ext = BlindExtractor(CKPT_DIR / "best_model.pth")
            b_res = b_ext.predict(atk_img_demo)
            blind_prob_demo = b_res["probability"]
            blind_bin_demo = b_res["binary"]
            blind_m_demo = score_against_ground_truth(blind_bin_demo, wm_gt)

        total_dt = time.perf_counter() - t_start
        prog.progress(1.0, text=f"Complete pipeline executed in {total_dt*1000:.0f} ms!")
        _log(f"Executed Complete Pipeline Demo ({demo_extractor_mode} | {demo_attack_choice}) in {total_dt:.2f}s")

        st.session_state["last_complete_demo"] = {
            "attack_choice": demo_attack_choice,
            "extractor_mode": demo_extractor_mode,
            "atk_img": atk_img_demo,
            "nb_bin": nb_out_demo["recovered_binary"],
            "nb_metrics": nb_m_demo,
            "blind_prob": blind_prob_demo,
            "blind_bin": blind_bin_demo,
            "blind_metrics": blind_m_demo,
            "total_dt": total_dt,
        }

    dstate = st.session_state["last_complete_demo"]
    use_blind = "Blind ANN" in demo_extractor_mode and dstate["blind_bin"] is not None
    active_bin = dstate["blind_bin"] if use_blind else dstate["nb_bin"]
    active_m = dstate["blind_metrics"] if use_blind else dstate["nb_metrics"]
    active_err = np.abs(active_bin.astype(np.float32) - wm_gt.astype(np.float32))

    if use_blind:
        st.success("🟢 **ACTIVE EXTRACTOR: BLIND ANN (`BlindWatermarkExtractor`)** — Extracted from the attacked image ONLY (no original host image, no crop mask).")
    else:
        st.warning("🟡 **ACTIVE EXTRACTOR: NON-BLIND (`NonBlindExtractor`)** — Uses the original clean host image and crop mask.")

    # Sequential stage cards
    st.markdown("#### Sequential Stage-by-Stage Execution View")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.image(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB), caption=f"Stage 1: Host YIQ\nI ∈ [{i_min:.1f}, {i_max:.1f}]", use_container_width=True)
    s2.image(_arr_to_png_bytes(wm_fwd["watermark_mosaic"]), caption=f"Stage 2: 8×8 Mosaic\n(ACM={acm_iters}, Cat={cat_iters})", use_container_width=True)
    s3.image(cv2.cvtColor(watermarked_bgr, cv2.COLOR_BGR2RGB), caption=f"Stage 3: Embedded RGB\nPSNR={emb_res['psnr']:.2f} dB", use_container_width=True)
    s4.image(_arr_to_png_bytes(dstate["atk_img"]), caption=f"Stage 4: Attacked\n{dstate['attack_choice']}", use_container_width=True)
    if use_blind and dstate["blind_prob"] is not None:
        s5.pyplot(_heatmap_fig(dstate["blind_prob"], "Stage 5: ANN Prob Map", cmap="magma", vmin=0, vmax=1))
    else:
        s5.image(_arr_to_png_bytes(dstate["nb_bin"], scale_nearest=256), caption="Stage 5: Tile Aggregation", use_container_width=True)
    s6.image(_arr_to_png_bytes(active_bin, scale_nearest=256), caption=f"Stage 6: Recovered\nNC={active_m['nc']:.4f}, BER={active_m['ber']:.4f}", use_container_width=True)

    st.markdown("#### Final Recovery Summary & Metrics")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Clean PSNR", f"{emb_res['psnr']:.2f} dB")
    k2.metric("Clean SSIM", f"{emb_res['ssim']:.4f}")
    k3.metric("Recovered NC", f"{active_m['nc']:.4f}")
    k4.metric("Recovered BER", f"{active_m['ber']:.4f}")
    k5.metric("Bit Accuracy", f"{active_m['accuracy']*100:.2f}%")
    k6.metric("Total Execution Time", f"{dstate['total_dt']*1000:.0f} ms")

    f1, f2, f3 = st.columns(3)
    f1.image(_arr_to_png_bytes(wm_gt, scale_nearest=256), caption="Original Ground-Truth Watermark (32×32)", use_container_width=True)
    f2.image(_arr_to_png_bytes(active_bin, scale_nearest=256), caption=f"Recovered Watermark ({'BLIND ANN' if use_blind else 'NON-BLIND'})", use_container_width=True)
    f3.pyplot(_heatmap_fig(active_err, f"Bit Error Map ({int(active_err.sum())} / 1024 bits wrong)", cmap="magma", vmin=0, vmax=1))

    dl1, dl2, dl3, dl4 = st.columns(4)
    dl1.download_button("⬇ Watermarked Image (PNG)", _arr_to_png_bytes(watermarked_bgr, is_bgr=True), "demo_watermarked.png", "image/png")
    dl2.download_button("⬇ Attacked Image (PNG)", _arr_to_png_bytes(dstate["atk_img"]), "demo_attacked.png", "image/png")
    dl3.download_button("⬇ Recovered Watermark (PNG)", _arr_to_png_bytes(active_bin, scale_nearest=256), "demo_recovered_watermark.png", "image/png")
    demo_json_bytes = json.dumps({
        "host": host_label,
        "extractor_mode": "BLIND_ANN" if use_blind else "NON_BLIND",
        "attack": dstate["attack_choice"],
        "psnr": emb_res["psnr"],
        "ssim": emb_res["ssim"],
        "recovery_metrics": active_m,
        "execution_seconds": dstate["total_dt"],
    }, indent=2).encode("utf-8")
    dl4.download_button("⬇ Full Results Report (JSON)", demo_json_bytes, "demo_results_report.json", "application/json")

    poster_p = BASE_DIR / "results" / "final_demo" / "full_pipeline.png"
    if poster_p.exists():
        with st.expander("View Generated 16-Panel Publication Overview (`results/final_demo/full_pipeline.png`)", expanded=False):
            st.image(str(poster_p), use_container_width=True)

with st.expander("📋 Execution Logs", expanded=False):
    st.text("\n".join(st.session_state["exec_logs"][-30:]) or "No stage actions logged yet.")
