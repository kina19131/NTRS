"""
merge_sigma_results.py
======================
Appends new sigma points from a second output directory into the base directory.
Keeps existing sigma points intact; only adds sigmas not already present.
Resorts by sigma after merging.

Usage:
  python merge_sigma_results.py --base ./outputs_0422_extended \
                                --new  ./outputs_0422_extended/extend2

After merging, re-run --replot to regenerate plots:
  python certified_density_experiment.py --replot \
      --output_dir ./outputs_0422_extended
"""

import argparse
import json
import os


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="Directory with existing JSON results (will be updated in-place)")
    p.add_argument("--new",  required=True,
                   help="Directory with new JSON results to merge in")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would change without writing")
    return p.parse_args()


def main():
    args = get_args()

    new_files = [f for f in os.listdir(args.new) if f.endswith(".json")]
    if not new_files:
        print(f"No JSON files found in {args.new}")
        return

    for fname in sorted(new_files):
        new_path  = os.path.join(args.new,  fname)
        base_path = os.path.join(args.base, fname)

        new_data = json.load(open(new_path))

        if not os.path.exists(base_path):
            # File doesn't exist in base — copy it wholesale
            print(f"  NEW (no base file):  {fname}")
            if not args.dry_run:
                with open(base_path, "w") as f:
                    json.dump(new_data, f, indent=2)
            continue

        base_data = json.load(open(base_path))

        existing_sigmas = {r["sigma"] for r in base_data["sigma_results"]}
        new_points      = [r for r in new_data["sigma_results"]
                           if r["sigma"] not in existing_sigmas]

        if not new_points:
            print(f"  SKIP (no new sigmas): {fname}")
            continue

        added_sigmas = sorted(r["sigma"] for r in new_points)
        print(f"  MERGE {fname}: adding σ = {added_sigmas}")

        if not args.dry_run:
            base_data["sigma_results"].extend(new_points)
            base_data["sigma_results"].sort(key=lambda r: r["sigma"])
            with open(base_path, "w") as f:
                json.dump(base_data, f, indent=2)

    if args.dry_run:
        print("\n(dry run — no files written)")
    else:
        print(f"\nDone. Re-run --replot to regenerate plots:")
        print(f"  python certified_density_experiment.py --replot --output_dir {args.base}")


if __name__ == "__main__":
    main()
