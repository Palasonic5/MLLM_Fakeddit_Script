# Fakeddit Zero-Shot Evaluation

Zero-shot multimodal fake news detection on the [Fakeddit](https://fakeddit.netlify.app/) dataset using vision-language models.

## Task

Given a Reddit post's **title** and **image**, classify it as:
- **2-way**: `real` (0) or `fake` (1)
- **6-way**: `true` (0), `satire` (1), `misleading` (2), `imposter` (3), `false` (4), `manipulated` (5)

Results are compared against ground-truth labels from the `multimodal_only_samples/` split.

---

## File Structure

```
eval/
├── dataloader.py            # Loads TSV splits, filters to samples with images
├── prompts.py               # Zero-shot prompt templates + response parser
├── metrics.py               # Accuracy, F1, precision, recall, confusion matrix
├── llama_fakeddit_eval.py   # Eval script for Llama 3.2-11B Vision Instruct
├── run_llama_eval.sbatch    # SLURM job for Llama eval
└── outputs/
    ├── llama_results.csv    # Per-sample predictions
    └── plots/               # Confusion matrix PNGs
```

---

## Data Download

### 1. Text and metadata (TSV files)

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm?usp=sharing -O dataset/
```

### 2. Images (110 GB archive)

```bash
gdown 1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b
```

Extract (uses parallel bzip2 for speed — requires `lbzip2`):

```bash
tar --use-compress-program=lbzip2 -xf public_images.tar.bz2
```

Or submit as a SLURM job if extraction is too slow interactively:

```bash
sbatch /scratch/YOUR-NETID/Fakeddit/unzip_images.sbatch
```

---

## Data Requirements

Before running eval, ensure the following exist:
- `/scratch/YOUR-NETID/Fakeddit/dataset/multimodal_only_samples/multimodal_test_public.tsv`
- `/scratch/YOUR-NETID/Fakeddit/public_image_set/` (extracted from `public_images.tar.bz2`)

Images are matched to TSV rows by `id` column: `public_image_set/{id}.jpg`.

---

## Configuration

Credentials and paths are stored in `.env` (gitignored). Copy the example and fill in your values:

```bash
cp .env.example .env
```

`.env` fields:

| Variable | Description |
|---|---|
| `NETID` | Your HPC net ID (used in all scratch paths) |
| `HF_TOKEN` | Hugging Face access token |
| `SLURM_ACCOUNT` | SLURM billing account |

---

## Running

### Llama 3.2-11B Vision Instruct

```bash
source .env
sbatch --account=${SLURM_ACCOUNT} run_llama_eval.sbatch
```

Key arguments in the sbatch (edit as needed):

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `.../llama-3.2-11B-Vision-Instruct` | Path to model weights |
| `--test_tsv` | `.../dataset/multimodal_only_samples/multimodal_test_public.tsv` | Test split TSV |
| `--image_dir` | `.../public_image_set` | Directory of post images |
| `--output_path` | `.../outputs/llama_results.csv` | Where to save predictions |
| `--plot_dir` | `.../outputs/plots` | Where to save confusion matrix |
| `--label_type` | `2_way` | `2_way` or `6_way` |
| `--max_samples` | `None` (all) | Limit samples for quick runs |

### Quick test (100 samples)

Add `--max_samples 100` to the python command in the sbatch (already set by default).

---

## Adding a New Model

1. Create `<model>_fakeddit_eval.py` — copy `llama_fakeddit_eval.py` as a template.
2. Replace only the `load_model_and_processor` and `run_single_inference` functions with the new model's API.
3. Keep all imports from `dataloader`, `prompts`, and `metrics` unchanged.
4. Create a corresponding `run_<model>_eval.sbatch`.

---

## Output

**`outputs/<model>_results.csv`** — one row per sample with columns:
- `id`, `clean_title`, `label`, `image_path`
- `raw_response` — raw model output text
- `predicted_label` — parsed integer label (None if unparseable)
- `correct` — 1/0/None

**`outputs/plots/<model>_<label_type>_confusion_matrix.png`** — confusion matrix heatmap.

Metrics printed to stdout: Accuracy, F1 (macro + weighted), Precision, Recall, per-class report.

---

## Notes

- Verify the 6-way integer-to-label mapping in `dataloader.py` (`LABEL_6WAY`) matches the actual TSV values before running 6-way eval. Run `df['6_way_label'].value_counts()` to check.
- The model is loaded in `bfloat16` with `device_map="auto"`. A single GPU with ≥40 GB VRAM is sufficient for the 11B model.
