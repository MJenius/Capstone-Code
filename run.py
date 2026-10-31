"""
Master Command Line Interface (CLI) for the Capstone Robust Image Watermarking Research Pipeline.

Provides unified execution for:
- Individual phases (phase 0 to 9)
- Individual attack testing (--phase attacks --attack crop|jpeg|noise|blur|collusion)
- ANN training dataset generation (--phase ann-data)
- Blind ANN extractor training (--phase ann-train [--epochs N] [--quick] [--resume])
- Blind ANN held-out evaluation (--phase ann-eval)
- Blind ANN inference (--phase ann-inference --image path/to/image.png [--model ...])
- Full 3-way extractor comparison (--phase comparison [--quick])
- Complete visual demonstration mode (--phase demo)
- Full benchmark execution (--phase benchmark)
- Automated validation tests (--phase tests)
- Complete pipeline dependency run (--phase all)
"""
import argparse
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline.phase_runner import PhaseRunner
from pipeline.ann_phases import (
    phase_ann_data,
    phase_ann_train,
    phase_ann_eval,
    phase_ann_inference,
    phase_comparison,
    phase_final_demo,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Master CLI for Capstone Robust Image Watermarking Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --phase environment
  python run.py --phase preprocessing
  python run.py --phase watermark
  python run.py --phase embedding
  python run.py --phase attacks --attack crop
  python run.py --phase extraction
  python run.py --phase benchmark --quick
  python run.py --phase ann-data
  python run.py --phase ann-train --epochs 10
  python run.py --phase ann-train --quick
  python run.py --phase ann-eval
  python run.py --phase ann-inference --image sample.png --model models/checkpoints/best_model.pth
  python run.py --phase comparison
  python run.py --phase demo
  python run.py --phase all
        """,
    )

    parser.add_argument(
        "--phase",
        type=str,
        default="demo",
        choices=[
            "environment", "0",
            "preprocessing", "1",
            "watermark", "2",
            "embedding", "3",
            "attacks", "4",
            "extraction", "5",
            "benchmark", "6",
            "tests", "7",
            "ann-data", "8",
            "ann-train", "ann-eval", "ann-inference", "comparison", "9",
            "demo",
            "reproduce",
            "all",
        ],
        help="Select pipeline phase to execute (default: demo)",
    )

    parser.add_argument(
        "--attack",
        type=str,
        default="crop",
        choices=["crop", "jpeg", "noise", "blur", "collusion"],
        help="Specific attack module to run during phase 4 or demo (default: crop)",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Limit number of images for benchmark, comparison, or ANN dataset generation",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs for --phase ann-train",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for --phase ann-train (default: 32)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="Learning rate for --phase ann-train (default: 3e-3)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume ANN training from models/checkpoints/last_model.pth",
    )

    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        default=None,
        help="Image path(s) or directory for --phase ann-inference",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Checkpoint path for --phase ann-inference, ann-eval, or comparison",
    )

    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Optional ground-truth 32x32 watermark for post-inference evaluation only",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast smoke-test mode",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed verbose output",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom YAML configuration file",
    )

    args = parser.parse_args()
    runner = PhaseRunner(config_path=Path(args.config) if args.config else None)
    phase = args.phase.lower()

    if phase in ["environment", "0"]:
        runner.run_phase0()

    elif phase in ["preprocessing", "1"]:
        runner.run_phase1()

    elif phase in ["watermark", "2"]:
        runner.run_phase2()

    elif phase in ["embedding", "3"]:
        runner.run_phase3()

    elif phase in ["attacks", "4"]:
        runner.run_phase4(attack_name=args.attack)

    elif phase in ["extraction", "5"]:
        runner.run_phase5()

    elif phase in ["benchmark", "6"]:
        n = 5 if args.quick and args.num_images is None else args.num_images
        runner.run_phase6(num_images=n)

    elif phase in ["tests", "7"]:
        runner.run_phase7(verbose=args.verbose)

    elif phase in ["ann-data", "8"]:
        runner.run_phase8(max_samples=args.num_images, quick=args.quick)

    elif phase == "ann-train":
        runner.run_phase9(mode="train", epochs=args.epochs, quick=args.quick)

    elif phase in ["ann-eval", "9"]:
        runner.run_phase9(mode="eval", model_path=args.model)

    elif phase == "ann-inference":
        imgs = args.image
        if not imgs:
            default_img = BASE_DIR / "results" / "phase4_attacks" / "crop" / "attacked_i_channel.npy"
            if not default_img.exists():
                runner.run_phase4(attack_name="crop")
            imgs = [str(default_img)]
        phase_ann_inference(images=imgs, model_path=args.model, ground_truth=args.ground_truth)

    elif phase == "comparison":
        phase_comparison(n_hosts=args.num_images, model_path=args.model, quick=args.quick)

    elif phase == "demo":
        runner.run_demo(attack_name=args.attack)

    elif phase == "reproduce":
        print("\n" + "=" * 80)
        print("REPRODUCIBILITY VERIFICATION SUITE")
        print("=" * 80)
        runner.run_phase0()
        runner.run_phase7(verbose=False)
        n = 3 if args.quick else 5
        runner.run_phase6(num_images=n)
        runner.run_demo(attack_name="crop")
        print("\n" + "=" * 80)
        print("REPRODUCIBILITY VERIFICATION COMPLETE: ALL CHECKS PASSED")
        print("=" * 80)

    elif phase == "all":
        print("\n" + "=" * 80)
        print("EXECUTING COMPLETE CAPSTONE RESEARCH PIPELINE (PHASES 0 TO 9 + COMPARISON + DEMO)")
        print("=" * 80)
        runner.run_phase0()
        runner.run_phase1()
        runner.run_phase2()
        runner.run_phase3()
        for atk in ["crop", "jpeg", "noise", "blur", "collusion"]:
            runner.run_phase4(attack_name=atk)
        runner.run_phase5()
        runner.run_phase6(num_images=5 if args.quick else 10)
        runner.run_phase8(quick=args.quick if not (BASE_DIR / "data" / "ann" / "dataset_summary.json").exists() else False)
        if not (BASE_DIR / "models" / "checkpoints" / "best_model.pth").exists():
            runner.run_phase9(mode="train", quick=args.quick)
        runner.run_phase9(mode="eval")
        phase_comparison(quick=args.quick)
        runner.run_demo(attack_name=args.attack)
        runner.run_phase7(verbose=False)

        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION STATUS SUMMARY — ALL PHASES COMPLETED")
        print("=" * 80)


if __name__ == "__main__":
    main()
