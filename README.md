# BLooP: Zero-Shot Abstractive Summarization using Large Language Models with Bigram Lookahead Promotion

Abstractive summarization requires models to generate summaries that convey information in the source document. While large language models can generate summaries without fine-tuning, they often miss key details and include extraneous information. We propose BLooP (Bigram Lookahead Promotion), a simple training-free decoding intervention that encourages large language models (LLMs) to generate tokens that form bigrams from the source document. BLooP operates through a hash table lookup at each decoding step, requiring no training, fine-tuning, or model modification. We demonstrate improvements in ROUGE and BARTScore for [Llama‑3.1‑8B‑Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [Mistral‑Nemo‑Instruct‑2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407), and [Gemma‑2‑9B‑IT](https://huggingface.co/google/gemma-2-9b-it) on CNN/DM, CCSum, Multi-News, and SciTLDR. Human evaluation shows that BLooP significantly improves faithfulness without reducing readability.

---

## How BLooP works

At each generation step, BLooP checks whether the pair *(previous token, candidate token)* appears as a bigram in the input document. If so, it promotes that candidate by adjusting its logit by α. Promotion is constant across all matching candidates, so their relative order is unchanged. This yields an efficient, training‑free copy mechanism implemented with a hash‑table bigram cache during decoding.

---

## Installation

```bash
git clone https://github.com/varuniyer/BLooP.git
cd BLooP

python -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install .

# spaCy model used for POS analysis (analyze.py)
python -m spacy download en_core_web_trf

# NLTK sentence tokenizer (matches Dockerfile)
python -m nltk.download punkt_tab

# Logins (required)
huggingface-cli login
wandb login
```

For running on NRP/Nautilus Kubernetes, see: [NRP/Nautilus Kubernetes setup](https://github.com/varuniyer/nrp-k8s-setup-template).

---

## Running BLooP

### Single experiment

Use the provided script (defaults are set in `run.sh`):

```bash
export MODE=single_run
bash run.sh
```

Or call the entrypoint directly:

```bash
python main.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset cnndm --split validation --subsample 1.0 \
  --max_input_len 4096 --beam_width 12 \
  --alpha 3 \
```

Recommended settings from our sweeps:

- [Llama‑3.1‑8B‑Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct): α = `3`, beam = `12`
- [Mistral‑Nemo‑Instruct‑2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407): α = `4`, beam = `5`
- [Gemma‑2‑9B‑IT](https://huggingface.co/google/gemma-2-9b-it): α = `6`, beam = `4`

### Sweeps

Run a grid over α, beam width, etc. with Weights & Biases agents:

```bash
export MODE=parallel_sweep
bash run.sh
```

Configuration: `sweep_config.yml` (optimizes BARTScore probability).

### Collect results

Aggregate W&B tables and export CSVs:

```bash
export MODE=collect_results
bash run.sh
```

Outputs:

- `perf_data.csv`
- `cache_usage_data.csv`
- `promotion_effect_data.csv`

---

## Datasets

Configured in `dataset_info.toml` and loaded via `datasets`. Tested on CNN/DM, CCSum, Multi‑News, and SciTLDR (Abs/AIC/Full). Prompts are assembled via each model’s Hugging Face chat template.

---

## Repo structure

```
BLooP/
├─ main.py                    # Entry: inference, logging, evaluation
├─ logits_process.py          # BLooP logits processor (bi‑gram cache promotion)
├─ analyze.py                 # Cache usage, position histograms, POS tags (W&B)
├─ eval.py                    # ROUGE + BARTScore (per‑example + aggregate)
├─ collect_wandb_results.py   # Aggregate runs → CSVs
├─ wilcoxon_test.py           # FDR‑corrected significance tests
├─ utils.py                   # Data loading, prompts, caches, helpers
├─ args.py, dataset_info.toml # CLI + dataset registry
├─ run.sh, entrypoint.sh      # Single run / sweeps / collection
├─ sweep_config.yml           # W&B grid config
├─ pyproject.toml             # Pinned deps
└─ LICENSE                    # MIT
```

---

## License

Released under the MIT License (see `LICENSE`).

---

## Acknowledgments

We acknowledge grants from the National Science Foundation and Google to support this research. This work used resources available through the National Research Platform (NRP) at the University of California, San Diego. NRP has been developed, and is supported in part, by funding from National Science Foundation, from awards 1730158, 1540112, 1541349, 1826967, 2112167, 2100237, and 2120019, as well as additional funding from community partners.
