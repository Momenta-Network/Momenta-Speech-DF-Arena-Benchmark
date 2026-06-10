#!/usr/bin/env python3
"""Reproduce Momenta's Speech DF Arena metrics from the released score files.

For each dataset this script joins the score file (scores/<dataset>.txt) with
the Arena protocol file (protocol_files/<dataset>.csv) and computes the Equal
Error Rate (EER), plus accuracy and F1 at the EER threshold. It then reports
the pooled metrics (all utterances from all 14 datasets scored under a single
global threshold) and the simple average across datasets.

The EER implementation follows the standard ASVspoof / Speech DF Arena
convention: scores are countermeasure scores where higher means more likely
bonafide, the false rejection rate is computed over bonafide trials and the
false acceptance rate over spoof trials, and the EER is taken at the
threshold where the two rates are equal.

Usage:
    python3 compute_metrics.py [--protocol-dir protocol_files] [--score-dir scores]

Requires: numpy, pandas
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = [
    "In_The_Wild",
    "ASVspoof2019",
    "ASVspoof2021_LA",
    "ASVspoof_2021_DF",
    "ASVSpoof2024",
    "Fake_or_Real",
    "CodecFake",
    "ADD_2022_track1",
    "ADD_2022_track3",
    "ADD_2023_Round_1",
    "ADD_2023_Round_2",
    "DIFFADD",
    "LibriSeVoc",
    "SONAR",
]


def compute_det_curve(target_scores, nontarget_scores):
    """DET curve over bonafide (target) and spoof (nontarget) scores."""
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """EER and the threshold at which it is reached."""
    frr, far, thresholds = compute_det_curve(
        np.array(target_scores).astype(np.longdouble),
        np.array(nontarget_scores).astype(np.longdouble),
    )
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def evaluate(labels, scores):
    """EER, accuracy and F1 (at the EER threshold) for one set of trials."""
    bonafide = scores[labels == "bonafide"]
    spoof = scores[labels == "spoof"]
    eer, threshold = compute_eer(bonafide, spoof)

    predictions = np.where(scores >= threshold, "bonafide", "spoof")
    accuracy = float(np.mean(predictions == labels))

    tp = int(np.sum((predictions == "bonafide") & (labels == "bonafide")))
    fp = int(np.sum((predictions == "bonafide") & (labels == "spoof")))
    fn = int(np.sum((predictions == "spoof") & (labels == "bonafide")))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return eer, accuracy, f1


def load_dataset(protocol_dir: Path, score_dir: Path, name: str):
    protocol = pd.read_csv(protocol_dir / f"{name}.csv", dtype={"file_name": str})
    scores = pd.read_csv(
        score_dir / f"{name}.txt",
        sep=" ",
        header=None,
        names=["file_name", "cm_score"],
        dtype={"file_name": str},
    )
    merged = protocol.merge(scores, on="file_name", how="left")
    missing = merged["cm_score"].isna().sum()
    if missing:
        sys.exit(f"{name}: {missing} protocol entries have no score — aborting.")
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", default="protocol_files", type=Path)
    parser.add_argument("--score-dir", default="scores", type=Path)
    parser.add_argument("--output", default=None, type=Path,
                        help="optional CSV path for the results table")
    args = parser.parse_args()

    rows = []
    pooled_frames = []
    for name in DATASETS:
        merged = load_dataset(args.protocol_dir, args.score_dir, name)
        eer, acc, f1 = evaluate(
            merged["label"].to_numpy(), merged["cm_score"].to_numpy()
        )
        rows.append(
            {"dataset": name, "utterances": len(merged),
             "eer_pct": 100 * eer, "accuracy_pct": 100 * acc, "f1": f1}
        )
        pooled_frames.append(merged[["label", "cm_score"]])
        print(f"{name:20s}  n={len(merged):>9,}  EER={100*eer:6.3f}%  "
              f"Acc={100*acc:6.2f}%  F1={f1:.4f}")

    pooled = pd.concat(pooled_frames, ignore_index=True)
    eer, acc, f1 = evaluate(pooled["label"].to_numpy(), pooled["cm_score"].to_numpy())
    table = pd.DataFrame(rows)
    avg = table[["eer_pct", "accuracy_pct", "f1"]].mean()

    print("-" * 72)
    print(f"{'AVERAGE':20s}  n={table['utterances'].sum():>9,}  "
          f"EER={avg['eer_pct']:6.3f}%  Acc={avg['accuracy_pct']:6.2f}%  "
          f"F1={avg['f1']:.4f}")
    print(f"{'POOLED':20s}  n={len(pooled):>9,}  EER={100*eer:6.3f}%  "
          f"Acc={100*acc:6.2f}%  F1={f1:.4f}")

    if args.output:
        summary = pd.concat(
            [table,
             pd.DataFrame([
                 {"dataset": "AVERAGE", "utterances": table["utterances"].sum(),
                  "eer_pct": avg["eer_pct"], "accuracy_pct": avg["accuracy_pct"],
                  "f1": avg["f1"]},
                 {"dataset": "POOLED", "utterances": len(pooled),
                  "eer_pct": 100 * eer, "accuracy_pct": 100 * acc, "f1": f1},
             ])],
            ignore_index=True,
        )
        summary.to_csv(args.output, index=False, float_format="%.4f")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
