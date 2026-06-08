# Quishing Detection — Hybrid CNN + XGBoost

End-to-end pipeline for the research paper
**"QR Code Phishing ('Quishing') Detection Using Hybrid ML on Mobile Devices."**

## Folder layout

```
QR paper/
├── QR_All_benign.zip        # provided (640 MB)
├── QR_All_Malicious.zip     # provided (365 MB)
├── figures.zip              # provided (legacy figures)
└── content/quishing-detection/
    ├── config.yaml          # all knobs live here
    ├── requirements.txt
    ├── run_all.bat          # ONE-CLICK runner
    ├── data/                # extracted images + CSVs land here
    ├── models/              # trained XGB / CNN / TFLite / meta
    ├── results/             # probabilities, metrics, ablation CSV
    ├── figures/             # final PNG + PDF plots
    └── src/
        ├── utils.py
        ├── 01_extract_data.py
        ├── 02_build_dataset.py
        ├── 03_url_features.py
        ├── 04_train_xgb.py
        ├── 05_train_cnn.py
        ├── 06_train_hybrid.py
        └── 07_evaluate.py
```

## Easiest way to run it

> **Pilot run** (default: 50 000 benign + 50 000 malicious) finishes on a normal laptop
> CPU in ~20–40 min. The CSVs are large, so the slowest step is the very first
> ZIP extraction (~10–20 min on HDD, faster on SSD).

1. Open **Command Prompt** (or PowerShell).
2. `cd "C:\Users\USER\Desktop\QR paper\content\quishing-detection"`
3. Double-click **`run_all.bat`** or run it from the terminal.

That's it — the batch file creates a venv, installs dependencies, then runs
steps 1–7 in order.

## Scaling to the full dataset (paper-grade)

Open `config.yaml` and set:

```yaml
dataset:
  sample_per_class: null   # use ALL samples
```

Then re-run `run_all.bat`. The CNN step will be the bottleneck (~1–3 h CPU,
~10–20 min on a single GPU).

## Manual run, one step at a time

```bash
cd src
python 01_extract_data.py        # unpack QR_All_*.zip into ../data/
python 02_build_dataset.py       # label, sample, split 70/10/20
python 03_url_features.py        # 27 lexical URL features -> parquet
python 04_train_xgb.py           # XGBoost on URL features
python 05_train_cnn.py           # small CNN on QR PNGs (+ TFLite export)
python 06_train_hybrid.py        # logistic-regression meta on top
python 07_evaluate.py            # ROC, confusion matrix, ablation, figures
```

## Outputs

| File | Purpose |
|---|---|
| `models/xgb_url.joblib` | URL classifier |
| `models/cnn_qr.keras`   | Structural CNN |
| `models/cnn_qr.tflite`  | Mobile-ready quantised CNN (<5 MB) |
| `models/hybrid_meta.joblib` | Meta-learner (final hybrid) |
| `results/model_comparison.csv` | XGB vs CNN vs Hybrid AUC |
| `results/ablation.csv` | Precision / recall / F1 / AUC per variant |
| `figures/fig1_roc_curve.png` | Headline ROC figure |
| `figures/fig2_confusion_matrix.png` | Hybrid confusion matrix |
| `figures/fig3_feature_importance.png` | Top URL features |
| `figures/fig4_ablation.png` | Component ablation |
| `figures/fig5_model_comparison.png` | Summary bars |

## Mobile deployment (Android stub)

`models/cnn_qr.tflite` is what you embed in the Android app. The XGBoost model
can be exported with `clf.save_model("xgb_url.json")` and either:

- ported to Android via the `xgboost4j` JNI binding, or
- converted to ONNX (`onnxmltools.convert_xgboost`) and run with the ONNX
  Runtime Mobile package.

A reference Android scanner that wires CameraX → ML Kit barcode scanner →
TFLite + ONNX inference lives outside this Python project.

## Evasion-robustness tests (paper §5)

To stress-test the model against the Barracuda 2025 variants, add:

- **Split QR**: tile two halves of a malicious QR over a benign frame.
- **Nested QR**: paste a small malicious QR inside the "logo slot" of a larger
  benign one.
- **PDF-embedded QR**: render the malicious PNGs into PDFs with `reportlab`
  and re-extract with `pdf2image` + `pyzbar`.

Re-run `07_evaluate.py` on each adversarial test set to report degradation.

## Citation reminder

You mentioned "Trap4Phish 2025" in the project brief — I could not independently
verify that dataset exists at the Canadian Institute for Cybersecurity. Please
double-check the source name before citing it in the paper.
