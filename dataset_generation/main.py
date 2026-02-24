"""
main.py

Entry point for sycophancy dataset generation.

Usage:
    # Generate all categories
    python main.py

    # Generate specific categories only
    python main.py --categories math physics

    # Generate with a custom topic count (per category)
    python main.py --n 50

    # Disable checkpoint resumption (start fresh)
    python main.py --no-resume

    # Dry run: just check connectivity, don't generate
    python main.py --dry-run
"""

import argparse
import logging
import sys
import os
from typing import List

# Make sure imports resolve from the dataset_generation root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, DATASET_CONFIG, OUTPUT_DIR, OLLAMA_BASE_URL
from generators import MathGenerator, PhysicsGenerator, PoliticalGenerator, OpinionGenerator
from utils.model_client import ModelClient
from utils.io_utils import ensure_dir

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "generation.log") if os.path.exists(OUTPUT_DIR) else "generation.log"),
    ],
)
logger = logging.getLogger("main")


# ─── Registry ─────────────────────────────────────────────────────────────────

GENERATOR_REGISTRY = {
    "math":      MathGenerator,
    "physics":   PhysicsGenerator,
    "political": PoliticalGenerator,
    "opinion":   OpinionGenerator,
}


# ─── Connectivity Check ───────────────────────────────────────────────────────

def check_connectivity():
    """Verify all backends are reachable before starting generation."""
    logger.info("Checking backend connectivity...")
    all_ok = True

    seen_backends = set()
    for category, cfg in MODEL_CONFIG.items():
        key = (cfg["backend"], cfg.get("model_id", ""))
        if key in seen_backends:
            continue
        seen_backends.add(key)

        if cfg["backend"] == "ollama":
            client = ModelClient(
                backend="ollama",
                model_id=cfg["model_id"],
                ollama_base_url=OLLAMA_BASE_URL,
            )
            if client.ping():
                logger.info(f"  ✓ Ollama reachable ({OLLAMA_BASE_URL})")
            else:
                logger.error(f"  ✗ Ollama NOT reachable at {OLLAMA_BASE_URL}")
                logger.error("    → Start Ollama with: ollama serve")
                all_ok = False
                break  # One check is enough
        else:
            logger.info(f"  ✓ HuggingFace backend for {cfg['model_id']} (local, assumed ok)")

    return all_ok


def print_model_summary(categories: List[str]):
    """Print a table of which model handles which category."""
    print("\n" + "="*60)
    print("  DATASET GENERATION PLAN")
    print("="*60)
    print(f"  {'Category':<12} {'Backend':<14} {'Model':<25} {'Topics'}")
    print(f"  {'-'*12} {'-'*14} {'-'*25} {'-'*6}")
    for cat in categories:
        cfg = MODEL_CONFIG[cat]
        n = DATASET_CONFIG[cat]["n_topics"]
        print(f"  {cat:<12} {cfg['backend']:<14} {cfg['model_id']:<25} {n}")
    print(f"\n  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("="*60 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sycophancy dataset with structured (P, C, Q) triplets."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(GENERATOR_REGISTRY.keys()),
        default=list(GENERATOR_REGISTRY.keys()),
        help="Which categories to generate (default: all)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Override number of topics per category (default: from config)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check connectivity and print plan, then exit",
    )
    return parser.parse_args()


def main():
    ensure_dir(OUTPUT_DIR)

    # Re-init logging now that OUTPUT_DIR exists
    file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, "generation.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    args = parse_args()
    categories = args.categories
    resume = not args.no_resume

    print_model_summary(categories)

    if not check_connectivity():
        logger.error("Aborting due to connectivity issues.")
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry run complete. Exiting.")
        return

    # ── Run generators ────────────────────────────────────────────────────────
    results = {}
    for category in categories:
        n_topics = args.n if args.n is not None else DATASET_CONFIG[category]["n_topics"]
        GeneratorClass = GENERATOR_REGISTRY[category]

        logger.info(f"\n{'='*50}")
        logger.info(f"  Starting: {category.upper()} ({n_topics} topics → {n_topics * 6} samples)")
        logger.info(f"{'='*50}")

        try:
            generator = GeneratorClass(resume=resume)
            samples = generator.generate(n_topics=n_topics)
            results[category] = {"status": "success", "n_samples": len(samples)}
        except KeyboardInterrupt:
            logger.warning(f"\nInterrupted during {category}. Progress saved to checkpoint.")
            results[category] = {"status": "interrupted"}
            break
        except Exception as e:
            logger.error(f"  FAILED: {category} — {e}", exc_info=True)
            results[category] = {"status": "failed", "error": str(e)}

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  GENERATION SUMMARY")
    print("="*60)
    for cat, info in results.items():
        status = info["status"]
        n = info.get("n_samples", "—")
        icon = "✓" if status == "success" else "✗"
        print(f"  {icon} {cat:<12} {status:<14} {n} samples")
    print(f"\n  Output saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()