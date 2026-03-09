"""Plot PER vs SNR for each language."""
import json
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    metrics_dir = Path(sys.argv[1])
    output_plot = Path(sys.argv[2])
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    # Collect all metrics
    results = {}
    for f in sorted(metrics_dir.rglob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        lang = data["lang"]
        snr = data["snr_db"]
        per = data["per"]
        if lang not in results:
            results[lang] = []
        results[lang].append((snr, per))

    plt.figure(figsize=(10, 6))

    all_points = {}
    for lang, points in sorted(results.items()):
        points.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 999), reverse=True)
        snrs = []
        pers = []
        for snr, per in points:
            label = "clean" if snr is None else snr
            snrs.append(label)
            pers.append(per)
            # For mean calculation
            if label not in all_points:
                all_points[label] = []
            all_points[label].append(per)

        plt.plot(range(len(snrs)), pers, marker="o", label=lang)

    # Mean curve
    if len(results) > 1:
        mean_snrs = []
        mean_pers = []
        for label in snrs:
            if label in all_points:
                mean_snrs.append(label)
                mean_pers.append(sum(all_points[label]) / len(all_points[label]))
        plt.plot(range(len(mean_snrs)), mean_pers, marker="s", linestyle="--", linewidth=2, label="mean", color="black")

    plt.xticks(range(len(snrs)), [str(s) for s in snrs])
    plt.xlabel("SNR (dB)")
    plt.ylabel("PER")
    plt.title("Phoneme Error Rate vs Noise Level")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"Plot saved to {output_plot}")


if __name__ == "__main__":
    main()