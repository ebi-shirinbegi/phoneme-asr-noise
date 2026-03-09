"""Add noise to audio files at a given SNR level."""
import json
import sys
import hashlib
import tempfile
import os
import numpy as np
import soundfile as sf
from pathlib import Path


def add_noise(signal, snr_db, rng):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def get_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    lang = sys.argv[1]
    snr_db = float(sys.argv[2])
    seed = int(sys.argv[3])
    input_manifest = Path(sys.argv[4])
    output_manifest = Path(sys.argv[5])

    out_wav_dir = Path(f"data/noisy/{lang}/snr_{int(snr_db)}/wav")
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    with open(input_manifest) as f:
        for line in f:
            entry = json.loads(line)
            signal, sr = sf.read(entry["wav_path"])
            if signal.ndim != 1:
                raise ValueError("Only mono audio supported")

            rng = np.random.default_rng(seed)
            noisy = add_noise(signal, snr_db, rng)

            stem = Path(entry["wav_path"]).stem
            noisy_path = str(out_wav_dir / f"{stem}.wav")
            sf.write(noisy_path, noisy, sr)

            new_entry = entry.copy()
            new_entry["wav_path"] = noisy_path
            new_entry["snr_db"] = snr_db
            new_entry["audio_md5"] = get_md5(noisy_path)
            entries.append(new_entry)

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

    print(f"Added noise (SNR={snr_db}dB) to {len(entries)} files")


if __name__ == "__main__":
    main()