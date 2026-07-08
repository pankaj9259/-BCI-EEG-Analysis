# Brain–Computer Interface Signals Analysis (ANN / ML)

Project to classify EEG signal patterns using the DEAP dataset. This repository provides scripts to download the DEAP preprocessed Python dataset, extract features (band powers + simple statistics), and train both classical ML and ANN-style models to classify arousal (low/high).

Setup

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell on Windows
pip install -r "d:/javoc project 2/requirements.txt"
```

2. Download DEAP preprocessed Python dataset (or download manually and place into `data/DEAP`):

```powershell
python src\preprocess.py --download --out data/DEAP
```

3. Extract features and save to NPZ file:

```powershell
python src\dataset.py --input data/DEAP --out data\features.npz
```

4. Train models:

```powershell
python src\train.py --features data\features.npz --out results
```

If the official DEAP download is unavailable in your environment, run the demo smoke path instead:

```powershell
python src\smoke.py
python src\train.py --features data\features_synthetic.npz --out results
```

Notes

- Example label: binary arousal using threshold 5 on DEAP ratings.
- Feature extraction uses Welch PSD bandpower for common EEG bands.
- `src/train.py` trains a RandomForest and an ANN-style scikit-learn MLP, then saves metrics and reports.
- If the DEAP host returns 503, the repository still includes a working demo pipeline so you can review the project structure and outputs immediately.
