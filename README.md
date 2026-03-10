# Phoneme ASR Robustness to Noise

Lab 3 — Professional Skills in Data Science and NLP  
Université Paris Cité, M1 Computational Linguistics  
Student: Mohammad Ebrahim SHARIFI

## Overview

A DVC pipeline that evaluates how noise affects phoneme recognition using `facebook/wav2vec2-lv-60-espeak-cv-ft` on Multilingual LibriSpeech data (French and German).

## Pipeline stages

1. **Download** — fetch audio from MLS test split, create JSONL manifests
2. **Phonemize** — convert text to IPA phonemes with espeak-ng
3. **Add noise** — white Gaussian noise at 10 SNR levels (30 to -15 dB)
4. **Predict** — run wav2vec2 phoneme recognizer
5. **Evaluate** — compute Phoneme Error Rate (PER)
6. **Plot** — PER vs noise for each language + mean

## Adding a new language

Only edit `params.yaml`:
```yaml
  es:
    mls_name: spanish
    espeak_voice: es
```

No code changes needed.

## Setup
```bash
pixi install
brew install espeak-ng
export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
```

## Run
```bash
pixi run python src/download_data.py fr
pixi run python src/phonemize.py fr data/manifests/fr/clean.jsonl data/manifests/fr/phonemized.jsonl
pixi run python src/run_all_snr.py fr
pixi run python src/plot.py data/metrics data/figures/per_vs_snr.png
```
