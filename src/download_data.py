"""Download MLS data and create clean manifests."""
import json
import hashlib
import sys
import tempfile
import os
from pathlib import Path
import yaml
from datasets import load_dataset
import soundfile as sf


def get_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    lang = sys.argv[1]

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    n_utterances = params["n_utterances"]
    seed = params["seed"]
    mls_name = params["languages"][lang]["mls_name"]

    out_wav_dir = Path(f"data/raw/{lang}/wav")
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(f"data/manifests/{lang}/clean.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading test split for {mls_name}...")
    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/facebook/multilingual_librispeech/{mls_name}/test-*.parquet",
        split="train",
    )
    ds = ds.shuffle(seed=seed).select(range(min(n_utterances, len(ds))))

    entries = []
    for i, item in enumerate(ds):
        audio = item["audio"]
        stem = f"mls_{item['id']}"
        utt_id = f"{lang}_{stem}"
        wav_path = f"data/raw/{lang}/wav/{stem}.wav"

        sf.write(str(out_wav_dir / f"{stem}.wav"), audio["array"], audio["sampling_rate"])

        entries.append({
            "utt_id": utt_id,
            "lang": lang,
            "wav_path": wav_path,
            "ref_text": item["transcript"],
            "ref_phon": None,
            "sr": audio["sampling_rate"],
            "duration_s": round(len(audio["array"]) / audio["sampling_rate"], 2),
            "snr_db": None,
            "audio_md5": get_md5(wav_path),
        })

        print(f"  [{i+1}/{n_utterances}] {utt_id}")

    # Atomic write
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=manifest_path.parent, suffix=".tmp", delete=False
    )
    try:
        for e in entries:
            tmp.write(json.dumps(e, ensure_ascii=False) + "\n")
        tmp.close()
        os.replace(tmp.name, manifest_path)
    except:
        tmp.close()
        os.unlink(tmp.name)
        raise

    print(f"Saved {len(entries)} utterances for '{lang}'")


if __name__ == "__main__":
    main()