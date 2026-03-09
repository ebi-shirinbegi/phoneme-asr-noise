"""Run phoneme recognition on audio files."""
import json
import sys
import tempfile
import os
import torch
import torchaudio
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def main():
    input_manifest = Path(sys.argv[1])
    output_manifest = Path(sys.argv[2])
    model_name = sys.argv[3]
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {model_name}...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()

    target_sr = 16000

    entries = []
    with open(input_manifest) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        entry = json.loads(line)
        waveform, sr = torchaudio.load(entry["wav_path"])

        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        # Make mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        inputs = processor(waveform.squeeze().numpy(), sampling_rate=target_sr, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        pred_phon = processor.decode(predicted_ids[0])

        entry["pred_phon"] = pred_phon
        entries.append(entry)
        print(f"  [{i+1}/{len(lines)}] {entry['utt_id']}: {pred_phon[:40]}...")

    # Atomic write
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=output_manifest.parent, suffix=".tmp", delete=False
    )
    try:
        for e in entries:
            tmp.write(json.dumps(e, ensure_ascii=False) + "\n")
        tmp.close()
        os.replace(tmp.name, output_manifest)
    except:
        tmp.close()
        os.unlink(tmp.name)
        raise

    print(f"Predicted {len(entries)} utterances")


if __name__ == "__main__":
    main()