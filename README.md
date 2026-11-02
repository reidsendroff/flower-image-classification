# Flower Classification (VGG16 Transfer Learning + Grad-CAM)

Notebook: `transfer-learning.ipynb`

## Provenance
This is my own university coursework, published here with permission to publish it. It is included as a portfolio sample.

## What it does
- Uses a **pretrained VGG16** (torchvision `VGG16_Weights.DEFAULT`, ImageNet) on the 5-class flower photos dataset.
- Stage 1 trains the new classifier head with the convolution layers frozen; stage 2 fine-tunes the last convolution block.
- Applies **Grad-CAM** to a test image using the stage 2 model.

## Dependencies
- Dataset: `flower_photos`, downloaded by the notebook into `./data` from the TensorFlow example-images URL in the notebook.
- Pretrained weights: downloaded by torchvision at run time.
- Python packages: see `requirements.txt` (unpinned).

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook transfer-learning.ipynb
```
Internet access is needed on first run for the dataset and weights. A GPU is optional but training on CPU will be slow.

## Status
- **No fresh end-to-end run has been done for this publication, and reproducibility has not been verified.**
- Any figures quoted in the notebook's markdown come from the original coursework run and have not been re-checked here.
- The repository contains no data, outputs, weights, or secrets.
