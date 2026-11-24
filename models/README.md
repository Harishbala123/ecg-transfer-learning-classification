# Models

| File | In Git? | Description |
|---|---|---|
| `alexnet_apnea.keras` | yes (~19 MB) | Trained classifier (inference-only, no optimizer state). Input: 64×128×3 RR/EDR wavelet image; output: P(normal), P(apnea) |
| `alexnet_imagenet_conv.npz` | no | ImageNet-1k AlexNet conv1–conv5 weights in Keras layout, needed only to **re-train**. Generated from torchvision `AlexNet_Weights.IMAGENET1K_V1` by `scripts/export_alexnet_weights.py`, which also verifies that the Keras conv stack reproduces torchvision's conv5 outputs |

Generate the pretrained weights (PyTorch CPU build is enough; Colab already has PyTorch):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python scripts/export_alexnet_weights.py
```
The pretrained ImageNet weights are not redistributed here; torchvision downloads them from download.pytorch.org.
