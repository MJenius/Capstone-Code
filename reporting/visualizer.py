"""
Reporting and visualization utilities for the Capstone pipeline.
Generates publication-quality, presentation-ready multi-panel figures and Markdown reports.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_image_grid(
    images_with_titles: List[Tuple[str, np.ndarray]],
    output_path: Path,
    cols: int = 3,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "gray",
) -> None:
    """
    Save a grid of images with titles using Matplotlib.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(images_with_titles)
    rows = (n + cols - 1) // cols

    if figsize is None:
        figsize = (cols * 4, rows * 4)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    for idx, (panel_title, img) in enumerate(images_with_titles):
        ax = axes[idx]
        if img.ndim == 3 and img.shape[2] == 3:
            # Assume RGB for matplotlib if uint8 or float in [0, 1]
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    if title:
        plt.suptitle(title, fontsize=14, fontweight="heavy", y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def create_demo_poster(
    host_bgr: np.ndarray,
    watermark_binary: np.ndarray,
    watermark_scrambled: np.ndarray,
    watermark_catalan: np.ndarray,
    watermark_mosaic: np.ndarray,
    watermarked_preview_bgr: np.ndarray,
    diff_heatmap: np.ndarray,
    attacked_preview: np.ndarray,
    recovered_mosaic: np.ndarray,
    recovered_catalan: np.ndarray,
    recovered_binary: np.ndarray,
    diff_watermark: np.ndarray,
    output_path: Path,
    metrics: Dict[str, Any],
) -> None:
    """
    Create a 12-panel high-resolution capstone demonstration poster suitable for PPT slides.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(18, 14))

    # Helper for BGR to RGB
    def to_rgb(bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    panels = [
        # Row 1: Host & Scrambling Pipeline
        ("1. Original Host (256x256)", to_rgb(host_bgr), None),
        ("2. Binary Watermark (32x32)", watermark_binary, "gray"),
        ("3. Arnold Scrambled (10 iters)", watermark_scrambled, "gray"),
        ("4. Catalan Permuted (5 iters, k=7)", watermark_catalan, "gray"),

        # Row 2: Mosaic, Embedding, & Attack
        ("5. 8x8 Mosaic (256x256, 64 tiles)", watermark_mosaic, "gray"),
        (f"6. Watermarked Host\nPSNR: {metrics.get('psnr', 0):.2f} dB, SSIM: {metrics.get('ssim', 0):.4f}", to_rgb(watermarked_preview_bgr), None),
        ("7. Perceptual Change Map (8x ampl.)", to_rgb(diff_heatmap), None),
        (f"8. Attacked Image ({metrics.get('attack_name', 'Attack')})", to_rgb(attacked_preview) if attacked_preview.ndim == 3 else attacked_preview, None if attacked_preview.ndim == 3 else "gray"),

        # Row 3: Extraction & Reversibility
        ("9. Extracted Residual Mosaic", recovered_mosaic, "gray"),
        ("10. Extracted Catalan Watermark", recovered_catalan, "gray"),
        (f"11. Recovered Watermark\nNC: {metrics.get('nc', 0):.4f}, BER: {metrics.get('ber', 0):.4f}", recovered_binary, "gray"),
        (f"12. Bit Error Map\nErrors: {metrics.get('bit_errors', 0)} / {watermark_binary.size}", diff_watermark, "magma"),
    ]

    for idx, (title, img, cmap) in enumerate(panels):
        r = idx // 4
        c = idx % 4
        ax = axes[r, c]
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    main_title = (
        "Capstone Research Pipeline: Hybrid Robust Watermarking Demonstration\n"
        f"Non-Blind Baseline | Image: {metrics.get('image_id', 'Sample')} | "
        f"Recovery Status: {'EXACT MATCH (100%)' if metrics.get('ber', 1) == 0.0 else 'ROBUST RECOVERY'}"
    )
    plt.suptitle(main_title, fontsize=14, fontweight="heavy", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_markdown_report(report_data: Dict[str, Any], output_path: Path) -> None:
    """Generate a clean, structured Markdown audit report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {report_data.get('title', 'Phase Report')}\n"]
    lines.append(f"**Generated:** {report_data.get('timestamp', 'N/A')}\n")
    lines.append(f"**Phase:** {report_data.get('phase', 'N/A')}\n")

    if "summary" in report_data:
        lines.append("## Executive Summary\n")
        lines.append(f"{report_data['summary']}\n")

    if "metrics" in report_data:
        lines.append("## Numerical Metrics\n")
        lines.append("| Metric | Value | Description |")
        lines.append("| :--- | :--- | :--- |")
        for k, v in report_data["metrics"].items():
            desc = v.get("desc", "") if isinstance(v, dict) else ""
            val = v.get("val", v) if isinstance(v, dict) else v
            lines.append(f"| **{k}** | `{val}` | {desc} |")
        lines.append("\n")

    if "artifacts" in report_data:
        lines.append("## Generated Artifacts\n")
        for item in report_data["artifacts"]:
            lines.append(f"- [{Path(item).name}](file:///{item})")
        lines.append("\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
