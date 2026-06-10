# Momenta — Speech DF Arena Benchmark Results

**0.89% pooled EER and 1.01% average EER across 2,341,286 audio samples and 14 independent deepfake-detection benchmarks** — the lowest error of any system, commercial or academic, measured against the [Speech DF Arena](https://huggingface.co/spaces/Speech-Arena-2025/Speech-DF-Arena) leaderboard.

This repository contains the complete, raw evidence behind those numbers: Momenta's per-utterance score files for all 14 Speech DF Arena evaluation datasets, the Arena protocol files they are scored against, and a self-contained script that recomputes every metric. Nothing here requires trusting us — clone the repo, run one command, and verify the results yourself.

📖 **Full analysis:** [Audio Deepfake Detection on the Speech DF Arena](https://momenta.network/blog/audio-deepfake-detection-2026) (Momenta blog)

> **Note:** The Speech DF Arena has temporarily paused its submission round while it prepares its next phase, so these results are not yet reflected on the public leaderboard. Competitor figures referenced in the blog post were taken from the live leaderboard as of June 2026. This repository puts Momenta's result on the record with full, reproducible methodology.

## Results

Computed by [`compute_metrics.py`](compute_metrics.py) from the files in this repository, using the Arena's published EER implementation. Accuracy and F1 are evaluated at the EER threshold. Lower EER is better.

| Dataset | Utterances | EER (%) | Accuracy (%) | F1 |
|---|---:|---:|---:|---:|
| In-the-Wild | 31,779 | 1.278 | 98.72 | 0.990 |
| ASVspoof 2019 (LA eval) | 71,237 | 0.408 | 99.59 | 0.981 |
| ASVspoof 2021 LA | 148,176 | 1.140 | 98.86 | 0.946 |
| ASVspoof 2021 DF | 533,928 | 0.309 | 99.69 | 0.947 |
| ASVspoof 5 (2024 eval) | 680,774 | 0.771 | 99.23 | 0.981 |
| Fake-or-Real | 4,528 | 0.972 | 99.05 | 0.991 |
| CodecFake | 304,241 | 0.860 | 99.14 | 0.986 |
| ADD 2022 Track 1 | 109,199 | 5.179 | 94.82 | 0.913 |
| ADD 2022 Track 3 | 126,861 | 0.491 | 99.51 | 0.985 |
| ADD 2023 Round 1 | 111,976 | 0.710 | 99.29 | 0.995 |
| ADD 2023 Round 2 | 118,477 | 0.833 | 99.17 | 0.994 |
| DFADD | 3,755 | 0.150 | 99.81 | 0.995 |
| LibriSeVoc | 92,407 | 0.318 | 99.68 | 0.989 |
| SONAR | 3,948 | 0.658 | 99.32 | 0.994 |
| **Average** | 2,341,286 | **1.005** | 98.99 | 0.978 |
| **Pooled** | 2,341,286 | **0.892** | 99.11 | 0.980 |

**Average EER** computes the error on each dataset separately and averages the 14 results. **Pooled EER** is stricter and closer to real-world deployment: it merges every utterance from all 14 datasets into a single pool and finds one global decision threshold that must work everywhere at once. A low pooled EER — lower here than the average EER — is the strongest evidence that a single deployed model generalizes across attack types (TTS, voice conversion, neural codecs, diffusion-based generators, and in-the-wild fakes) without per-dataset tuning.

The full results table is also available as [`results/metrics.csv`](results/metrics.csv).

## Reproduce the results

Requirements: Python 3.9+, `numpy`, `pandas`.

```bash
git clone https://github.com/Momenta-Network/Momenta-Speech-DF-Arena-Benchmark.git
cd Momenta-Speech-DF-Arena-Benchmark
pip install numpy pandas
python3 compute_metrics.py
```

This prints per-dataset EER / accuracy / F1, the 14-dataset average, and the pooled metrics, all computed directly from `scores/` and `protocol_files/`. Pass `--output results/metrics.csv` to also write the table to CSV. The run takes under a minute on a laptop.

The EER implementation in `compute_metrics.py` follows the Speech DF Arena's official scoring code ([`speech_df_arena/utils/metrics.py`](https://github.com/Speech-Arena/speech_df_arena/blob/main/utils/metrics.py)), which in turn follows the standard ASVspoof convention: the false rejection rate is measured over bonafide trials, the false acceptance rate over spoof trials, and the EER is taken at the threshold where the two are equal.

## Repository contents

```
.
├── protocol_files/        # Speech DF Arena evaluation protocols (one CSV per dataset)
│   └── <dataset>.csv      #   columns: file_name,label  (label ∈ {bonafide, spoof})
├── scores/                # Momenta's per-utterance scores (one TXT per dataset)
│   └── <dataset>.txt      #   format: "<file_name> <score>", space-separated, no header
├── results/
│   └── metrics.csv        # computed metrics table (output of compute_metrics.py)
├── compute_metrics.py     # reproduces every number in the table above
└── README.md
```

### Score file format

Each line of a score file pairs an utterance identifier with a single scalar countermeasure score produced by Momenta's audio deepfake detection model:

```
LA_E_2834763 -5.192282676696777
LA_E_8877452 -5.1425323486328125
```

**Higher scores indicate bonafide (genuine) speech; lower scores indicate spoofed/synthetic speech.** Scores are raw model outputs — no per-dataset calibration, no per-dataset thresholds. Every utterance in every protocol file has a corresponding score; `compute_metrics.py` asserts this before computing anything.

### Protocol files

The protocol files are the Speech DF Arena's published evaluation protocols (file lists and bonafide/spoof labels) for its 14 test sets, included here so the metrics are reproducible from this repository alone. They contain utterance identifiers and labels only — **no audio is distributed in this repository**. The underlying audio can be obtained from each dataset's original distribution, linked from the [Speech DF Arena toolkit](https://github.com/Speech-Arena/speech_df_arena).

Two file-naming notes: `DIFFADD` corresponds to the dataset published as **DFADD**, and `ASVSpoof2024` corresponds to the **ASVspoof 5** evaluation set. Names follow the Arena's protocol-file naming.

## About the benchmark

[Speech DF Arena](https://huggingface.co/spaces/Speech-Arena-2025/Speech-DF-Arena) is the first comprehensive, independent benchmark and public leaderboard for speech deepfake detection, built by researchers from the Idiap Research Institute, MBZUAI, Tallinn University of Technology, and partner institutions. It evaluates every system — open-source and commercial — on the same 14 datasets with the same scoring code, making results directly comparable. Systems are ranked by pooled EER.

The 14 evaluation sets span the full spectrum of modern audio deepfake attacks:

- **TTS and voice-conversion attacks** — ASVspoof 2019 / 2021 LA / 2021 DF / ASVspoof 5, Fake-or-Real, SONAR
- **Neural codec artifacts** — CodecFake, LibriSeVoc
- **Diffusion-based generators** — DFADD
- **Multilingual challenge sets (English and Chinese)** — ADD 2022 / 2023
- **Real-world deepfakes mined from social media** — In-the-Wild

Benchmark paper:

```bibtex
@article{kulkarni2026speechdfarena,
  title   = {Speech DF Arena: A Leaderboard for Speech DeepFake Detection Models},
  author  = {Kulkarni, Ajinkya and Kulkarni, Atharva and Dowerah, Sandipana and others},
  journal = {IEEE Open Journal of Signal Processing},
  year    = {2026},
  doi     = {10.1109/OJSP.2026.3652496},
  note    = {arXiv:2509.02859}
}
```

## About Momenta

[Momenta](https://momenta.network) builds real-time deepfake detection and social-engineering simulation for enterprises. The scores in this repository were produced by the same audio detection model that powers [Momenta Detection](https://momenta.network/product/detection), which analyzes audio and video streams in milliseconds and produces a risk score before a decision is acted on.

- 🌐 Website: [momenta.network](https://momenta.network)
- 📖 Benchmark deep-dive: [Audio Deepfake Detection on the Speech DF Arena](https://momenta.network/blog/audio-deepfake-detection-2026)
- 🛡️ Products: [Detection](https://momenta.network/product/detection) · [Simulation](https://momenta.network/product/simulation)

## License and attribution

- Momenta's score files (`scores/`), the evaluation script, and the results tables are released under the [MIT License](LICENSE) — use them freely with attribution.
- The protocol files originate from the [Speech DF Arena](https://github.com/Speech-Arena/speech_df_arena) and the underlying datasets (ASVspoof, ADD, In-the-Wild, CodecFake, DFADD, LibriSeVoc, Fake-or-Real, SONAR); labels remain subject to the terms of their original distributions. They are redistributed here, without any audio, solely to make the benchmark results verifiable.

If you use these results or files in your work, please cite the Speech DF Arena paper above and link to this repository or the [blog post](https://momenta.network/blog/audio-deepfake-detection-2026).
