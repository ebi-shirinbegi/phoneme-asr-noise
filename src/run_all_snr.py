"""Run noise + predict + evaluate for all SNR levels."""
import subprocess
import sys
import yaml

def main():
    lang = sys.argv[1]

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    snr_levels = params["snr_levels"]
    seed = params["seed"]
    model = params["model_name"]

    input_manifest = f"data/manifests/{lang}/phonemized.jsonl"

    for snr in snr_levels:
        print(f"\n=== SNR={snr}dB ===")
        noisy_manifest = f"data/manifests/{lang}/noisy_{snr}.jsonl"
        pred_manifest = f"data/manifests/{lang}/predictions_{snr}.jsonl"
        metric_file = f"data/metrics/{lang}/snr_{snr}.json"

        subprocess.run(["python", "src/add_noise.py", lang, str(snr), str(seed), input_manifest, noisy_manifest], check=True)
        subprocess.run(["python", "src/predict.py", noisy_manifest, pred_manifest, model], check=True)
        subprocess.run(["python", "src/evaluate.py", pred_manifest, metric_file], check=True)

    # Also run clean
    print("\n=== CLEAN ===")
    subprocess.run(["python", "src/predict.py", input_manifest, f"data/manifests/{lang}/predictions_clean.jsonl", model], check=True)
    subprocess.run(["python", "src/evaluate.py", f"data/manifests/{lang}/predictions_clean.jsonl", f"data/metrics/{lang}/clean.json"], check=True)


if __name__ == "__main__":
    main()