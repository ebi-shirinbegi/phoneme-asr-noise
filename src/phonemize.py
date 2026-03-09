"""Add phoneme references to a manifest using espeak-ng."""
import json
import sys
import subprocess
import tempfile
import os
from pathlib import Path

LANG_MAP = {
    "fr": "fr-fr",
    "de": "de",
}


def phonemize(text, lang):
    result = subprocess.run(
        ["espeak-ng", "-v", lang, "-q", "--ipa", text],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main():
    lang = sys.argv[1]
    input_manifest = Path(sys.argv[2])
    output_manifest = Path(sys.argv[3])
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    espeak_lang = LANG_MAP[lang]

    entries = []
    with open(input_manifest) as f:
        for line in f:
            entry = json.loads(line)
            entry["ref_phon"] = phonemize(entry["ref_text"], espeak_lang)
            entries.append(entry)
            print(f"  {entry['utt_id']}: {entry['ref_phon'][:40]}...")

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

    print(f"Phonemized {len(entries)} utterances")


if __name__ == "__main__":
    main()