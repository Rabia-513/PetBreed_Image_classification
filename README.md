# 🐾 PetBreed AI — 37-Breed Image Classification 

**PyTorch · Torchvision · Transfer Learning · Kaggle · Gradio**

![PetBreed AI dashboard](![alt text](image.png)

A practical AI engineering project for classifying **37 cat and dog breeds** using the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). Three ImageNet-pretrained convolutional networks were fine-tuned and compared on a fixed validation split. The selected model, **EfficientNet-B0**, was then evaluated on the held-out Oxford test set and on images sourced from a different repository.

> **Main finding:** EfficientNet-B0 achieved **90.24% accuracy** on the official Oxford test set, but only **53.66% preliminary accuracy (22/41)** on a smaller, differently sourced **cat-only** sample. The external dataset has provisional breed labels and has **not** been conclusively checked for overlap with Oxford-IIIT Pet, so this difference is descriptive, not a definitive distribution-shift estimate.

## Project workflow

`Oxford-IIIT Pet → Stratified train/validation split → Preprocessing → Fine-tune three models → Compare validation macro-F1 → Select EfficientNet-B0 → Official test evaluation → External image evaluation → Gradio demonstration`

## Results at a glance

### Architecture selection — validation set (736 images)

| Architecture | Validation accuracy | Validation macro-F1 |
|---|---:|---:|
| ResNet-18 | 92.12% | 92.13% |
| **EfficientNet-B0** | **93.07%** | **93.15%** |
| MobileNetV3-Large | 90.22% | 90.13% |

The model was selected by **validation macro-F1**, without using the official test set for architecture selection. These values are from the saved validation-evaluation outputs in the notebook; repeated training can vary slightly.

### Selected model — official Oxford test set (3,669 images)

| Metric | EfficientNet-B0 |
|---|---:|
| Accuracy | **90.24%** |
| Macro precision | 90.53% |
| Macro recall | 90.22% |
| Macro-F1 | 90.04% |

The notebook computes and displays a **37-class precision/recall/F1 report** and a **37 × 37 confusion matrix**. It also saves the per-class table under `results/efficientnet_test_per_class.csv` when executed.

**Three most frequent directional breed confusions on the official test set:**

| Actual breed | Predicted breed | Incorrect images |
|---|---|---:|
| American Pit Bull Terrier | Staffordshire Bull Terrier | 32 |
| Ragdoll | Birman | 15 |
| American Pit Bull Terrier | American Bulldog | 14 |

These are *directional* errors, rather than totals merged across both directions.

### Generalisation — external cat photographs

| Evaluation | Images | Correct | Top-1 accuracy |
|---|---:|---:|---:|
| Official Oxford test | 3,669 | — | 90.24% |
| External cat-only sample | 41 | 22 | **53.66%** |

**Observed difference: −36.58 percentage points.**

A separate [Cat Breeds Dataset](https://github.com/AtharvaTaras/Cat-Breeds-Dataset) was used to collect **45 candidate images** across nine overlapping cat breeds. Four unsuitable images (including illustrations/graphics and a collage) were excluded during visual screening, leaving **41** for preliminary inference. These images were **not used for training or fine-tuning**.

**Limitations:** The external sample covers nine cat breeds, whereas the Oxford test set includes 37 cat/dog breeds. Sample sizes per breed are small (approximately three to five). Breed identities rely on source-provided labels, and overlap/near-duplicate checks against Oxford are not yet complete. Accordingly, the external result is marked **preliminary** and is not a controlled, breed-matched measure of distribution shift. Further verification is required before claiming a fully independent benchmark.

## Example failure analysis

The external image `persian_01.jpg` was source-labeled **Persian**, but EfficientNet-B0 predicted **Maine Coon** with about **93.0% softmax score**. Both breeds can share long fur and some visible traits, so the model *may* be relying on coat texture or other features that do not discriminate them reliably in that image. This is a hypothesis, **not a demonstrated cause**. The source breed label must also be confirmed independently.

**What I would try next:** verify the source label and check for overlap; use Grad-CAM to inspect salient regions; gather additional independently labeled images of these breeds; and test targeted augmentations/regularisation **using a separate development set**, preserving an untouched external holdout for final comparison.

## Data splitting and preprocessing

The dataset's official **3,669-image test partition** was retained. The **3,680-image trainval partition** was split **80/20 with stratification and seed 42**:

| Split | Images | Role |
|---|---:|---|
| Training | 2,944 | Update model weights |
| Validation | 736 | Select checkpoint / architecture |
| Official test | 3,669 | Final Oxford benchmark |

The saved split is reproducible using `random_state=42`, and the notebook exports the train/validation indices to `results/data_split.csv`.

Training images use random resized crops, horizontal flips, small rotations, color jitter, tensor conversion and ImageNet normalization. Validation and test images use deterministic resize to 256 on the shorter side, center crop to 224 × 224, tensor conversion and ImageNet normalization. Batch size is 32.

## Model and training choices

All three architectures start from Torchvision ImageNet weights. Their final classification layers are adapted to output **37 logits**. The same broad procedure was used for each:

| Setting | Value |
|---|---|
| Loss | Cross-entropy |
| Optimizer | AdamW |
| Head-only stage | 3 epochs, learning rate `1e-3` |
| Full fine-tuning stage | Up to 10 epochs, learning rate `1e-4` |
| Weight decay | `1e-4` |
| Batch size | 32 |
| Early-stopping patience | 4 fine-tuning epochs |
| Checkpoint criterion | Best validation macro-F1 |
| Seed | 42 |

Training histories and model checkpoints are written under `/kaggle/working/results/` and `/kaggle/working/models/` during notebook execution. The three networks share a general training recipe but differ in architecture; equal hyperparameters do not imply equal compute or identical optimization dynamics.

## How to run

### Option A — Kaggle (recommended; reproduces the full notebook)

1. Upload **`internship-task.ipynb`** to a new [Kaggle Notebook](https://www.kaggle.com/code).
2. Enable a **GPU** in session settings. Enable internet access if downloads are needed.
3. Run the notebook **from top to bottom**, including dataset loading, model creation, training, validation, official test evaluation and external image collection/evaluation.
4. Inspect the final tables and plots. The notebook saves checkpoints and CSVs under `/kaggle/working/`.
5. Run the Gradio cell at the end, then open the URL printed by `demo.launch(...)` (public share links are temporary).
6. Save a Kaggle notebook version **with output files** and download/back up the saved model checkpoints. Draft-session `/kaggle/working/` contents can be lost when a session ends.

**To skip retraining:** Restore the three saved `*_best.pth` checkpoints in `/kaggle/working/models/` and ensure the notebook definitions (`build_model`, `eval_transform`, `class_names`, DataLoaders, etc.) are executed. **Do not run the three-model training loop again.** Run only the evaluation and Gradio cells you need. The current Gradio implementation lives **inside the notebook**, so it is not a stand-alone service independent of notebook state.

### Option B — Local environment

The notebook is written for Kaggle paths, so a local run requires changing paths such as `/kaggle/working/`, installing dependencies, and making the images/checkpoints available. A GPU is recommended for retraining. Main libraries used: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `Pillow`, `requests`, and `gradio`.

## Gradio interface and screenshots

The Gradio interface is implemented in the notebook. It supports dataset/preprocessing previews, on-demand model training, validation comparison, learning curves, official-test evaluation, upload/URL predictions, and external-image generalisation analysis. **Training starts only when the Start Training button is pressed**; saved checkpoints are used for inference.

The Gradio share URL generated by Kaggle is temporary, so the screenshots below document the interface even when the live session is offline.

### Model comparison and training curves

![Three-model validation comparison](![alt text](image-1.png),![alt text](image-2.png))

### Official test evaluation

![Official test confusion matrix and evaluation interface](![alt text](image-3.png))

### Upload-based breed prediction

![Pet image prediction in Gradio](![alt text](image-4.png))

### External generalisation

![External image evaluation dashboard](![alt text](image-5.png)  )

> **Screenshot setup:** Save your images in the `screenshots/` folder using the exact filenames in the links above. If you did not capture a particular screen, remove that heading/image pair from this README before publishing to avoid broken images. Screenshots document UI behavior; numerical results should be verified against the notebook outputs.

## Reproducibility and evaluation cautions

- The official test set is held out from validation-based architecture selection.
- This is a notebook-based research demonstration, not a production breed-identification service.
- The external images use provisional web-source labels; **independent breed-label and overlap verification remain unfinished**.
- Softmax scores are not necessarily calibrated probabilities.
- Model training results may vary across hardware/software versions.
- To reproduce inference, include the checkpoint files or a reliable checkpoint download method. Do not commit oversized weights to Git by accident.
- The current notebook's UI includes optional training controls. Do not click them when reproducing inference only.

## Future improvements

1. Verify external breed labels and remove exact/near duplicates against Oxford images.
2. Expand external evaluation to dog breeds and conduct a breed-matched comparison.
3. Study confidently wrong predictions with Grad-CAM and calibration metrics.
4. Test separate, documented regularisation/augmentation experiments without tuning to the final test sets.
5. Package inference as a standalone `app.py` so the demo can run outside the notebook.

## Submission

The requested submission is a **public GitHub repository link**, not a ZIP or Drive link. The repository should contain the notebook, this README, and any distributable experiment artifacts or documented download steps needed to reproduce the reported results.

**Dataset attribution:** [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) · [External cat-image source](https://github.com/AtharvaTaras/Cat-Breeds-Dataset). Review original-image licenses before redistributing any external photographs in the public repository.
