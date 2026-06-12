"""Pre-bake tag-dropout caption variants as .txt lines (cached augmentation).

Writes K samples of the tag-dropout distribution as one caption per line, so
training can keep ``cache_text_embeddings = true`` (the text encoder stays off
the GPU) while still getting dropout regularization: every variant's embedding
is cached and the epoch accounting rotates them (one pass over the images per
epoch; see docs/developer/dataset-and-cache.md "Caption variants").

The base caption is the image's current first .txt line; with --in-place the
original file is backed up once to <name>.txt.orig and reused as the source on
later runs, so regenerating with a new K/seed/probability is idempotent.

Usage:
  python scripts/generate_caption_variants.py DATASET_DIR --variants 15 --seed 42 \
      [--probability 0.3] [--mode per_tag|full] [--out OTHER_DIR | --in-place]

Set tag_dropout_enabled = false in the dataset TOML afterwards — the dropout is
baked into the lines, and live dropout is rejected with cached embeddings.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rengu_flow.data.tag_dropout import TagDropoutConfig, apply_tag_dropout

IMAGE_SUFFIXES = (".webp", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--variants", "-k", type=int, default=15,
                        help="Variants per image; epochs many is statistically sufficient (default 15).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probability", type=float, default=0.3,
                        help="Per-tag drop probability (default 0.3).")
    parser.add_argument("--mode", choices=("per_tag", "full"), default="per_tag")
    parser.add_argument("--shuffle-tags", action="store_true",
                        help="Also shuffle tag order in each variant (replaces the retired "
                             "shuffle_tags/cache_shuffle_num cache-time shuffling).")
    parser.add_argument("--delimiter", default=", ", help="Tag delimiter (default ', ').")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path, help="Write a copy of the dataset here.")
    group.add_argument("--in-place", action="store_true",
                       help="Rewrite .txt files in place (originals backed up to .txt.orig once).")
    args = parser.parse_args()

    if args.probability <= 0 and not args.shuffle_tags:
        print("nothing to do: probability is 0 and --shuffle-tags is off", file=sys.stderr)
        return 1
    cfg = TagDropoutConfig(enabled=args.probability > 0, default_probability=args.probability, mode=args.mode)
    rng = random.Random(args.seed)
    images = sorted(
        p for p in args.dataset_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"no images found in {args.dataset_dir}", file=sys.stderr)
        return 1
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    written = 0
    line_counts = set()
    for img in images:
        txt = img.with_suffix(".txt")
        orig = txt.with_suffix(".txt.orig")
        source = orig if (args.in_place and orig.exists()) else txt
        if not source.exists():
            print(f"skipping {img.name}: no caption file", file=sys.stderr)
            continue
        base_captions = [
            line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        # K variants of EACH base caption: a multi-caption .txt (alternative
        # descriptions) keeps every alternative, each with its own dropout
        # samples. Total lines = len(base_captions) * K.
        variants = []
        for caption in base_captions:
            for _ in range(args.variants):
                variant = apply_tag_dropout(caption, cfg, rng)
                if args.shuffle_tags:
                    parts = [t.strip() for t in variant.split(args.delimiter) if t.strip()]
                    rng.shuffle(parts)
                    variant = args.delimiter.join(parts)
                variants.append(variant)
        if args.in_place:
            if not orig.exists():
                shutil.copy2(txt, orig)
            txt.write_text("\n".join(variants) + "\n", encoding="utf-8")
        else:
            shutil.copy2(img, args.out / img.name)
            (args.out / txt.name).write_text("\n".join(variants) + "\n", encoding="utf-8")
        written += 1
        line_counts.add(len(variants))

    where = args.dataset_dir if args.in_place else args.out
    print(
        f"{written} images -> {where} with {args.variants} variants per base caption "
        f"(p={args.probability}, mode={args.mode}, seed={args.seed}). "
        "Remember: tag_dropout_enabled = false + cache_text_embeddings = true, then re-cache."
    )
    if len(line_counts) > 1:
        print(
            f"WARNING: images have differing caption-line counts ({sorted(line_counts)}); "
            "the epoch accounting only divides variants out when the count is uniform — "
            "mixed counts fall back to variant-inflated epochs.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
