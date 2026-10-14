# Potato tuber disease classification: CNN architectures × activation functions

Code and data for the revised manuscript *Systematic Evaluation of Deep Convolutional Neural Networks and Activation Functions for Potato Tuber Disease Classification: A Leakage-Controlled Cross-Validation Study*.

## What is here

| Path | Contents |
|---|---|
| `potato_rev/` | Training pipeline: data cache and GPU augmentation, activation replacement, training loop, metrics, resumable multi-process runner, inference benchmark |
| `tools/audit_dataset.py` | Duplicate audit and fold assignment; writes `data_audit/manifest.json` |
| `tools/analyze_results.py` | Learning-rate selection on validation data, tables, statistical tests, figures |
| `tools/gradcam.py` | Grad-CAM explanations for one configuration |
| `tools/make_method_figures.py`, `tools/build_manuscript.py` | Method figures; assembly of the manuscript documents (manuscript sources are not part of this repository) |
| `data_audit/manifest.json` | Curated image list (MD5 per file), duplicate groups, related-image groups, fold of every image |
| `revision_outputs/runs/` | One JSON per training run (configuration, learning curves, validation and test predictions) |
| `revision_outputs/analysis/` | Tables (CSV), figures (PNG) and `summary.txt` |
| `legacy/` | Code of the first manuscript version (single train/val/test split), kept for reference |

## Data

The images are the public Kaggle collection "Potato Diseases Datasets" (451 images, 7 classes), expected in `potato_disease_dataset/<class>/`. The pipeline checks every file against the MD5 in the manifest, so a different copy of the data is detected.

## Reproducing the experiments

```bash
python tools/audit_dataset.py --report --write        # duplicate audit and folds (already in data_audit/)
python -m potato_rev.runner --stages grid noaug scratch  # 1,260 runs; resumable; workers chosen from GPU memory
python -m potato_rev.runner --stages full100             # 100-epoch curves (needs the grid results)
python -m potato_rev.benchmark                           # inference cost of all 49 configurations
python tools/analyze_results.py                          # tables, statistics, figures
python tools/gradcam.py --arch densenet121 --act relu --lr 3e-4 --fold 0
python tools/make_method_figures.py
```

Paths can be redirected with the environment variables `POTATO_DATA_DIR`, `POTATO_MANIFEST` and `POTATO_OUT_DIR`. The original runs used a Colab A100 (40 GB) with six workers, about 9 GPU-hours in total. Requirements: Python 3.10+, PyTorch 2.x with torchvision, NumPy, Pillow; the analysis additionally needs pandas, SciPy and matplotlib, and the document builder python-docx.

## Protocol in brief

Five stratified folds (photographs of one specimen kept together). In rotation k, fold k is the test set, fold k+1 the validation set and the other three folds the training set. Every configuration is trained at four learning rates; the run with the lowest validation loss is evaluated once on the test fold. All runs use 32-bit precision and, within a rotation, the same seed for every configuration.
