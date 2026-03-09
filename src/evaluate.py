"""Compute Phoneme Error Rate (PER) from predictions."""
import json
import sys
import tempfile
import os
from pathlib import Path
from jiwer import cer


def main():
    input_manifest = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    output_file.parent.mkdir(parents=True, exist_ok=True)

    refs = []
    preds = []
    entries = []

    with open(input_manifest) as f:
        for line in f:
            entry = json.loads(line)
            ref = entry["ref_phon"]
            pred = entry.get("pred_phon", "")

            # Remove spaces from predicted (model outputs spaced phonemes)
            pred_clean = pred.replace(" ", "")

            refs.append(ref)
            preds.append(pred_clean)
            entries.append(entry)

    per = cer(refs, preds)

    result = {
        "lang": entries[0]["lang"],
        "snr_db": entries[0]["snr_db"],
        "per": round(per, 4),
        "n_utterances": len(entries),
    }

    # Atomic write
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=output_file.parent, suffix=".tmp", delete=False
    )
    try:
        tmp.write(json.dumps(result, indent=2))
        tmp.close()
        os.replace(tmp.name, output_file)
    except:
        tmp.close()
        os.unlink(tmp.name)
        raise

    print(f"PER = {result['per']} (lang={result['lang']}, snr={result['snr_db']})")


if __name__ == "__main__":
    main()