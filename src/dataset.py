"""Load DEAP preprocessed python files, extract EEG features, and save feature arrays.

Produces an .npz file with `X` (features) and `y` (binary arousal labels).
"""
import argparse
import os
import pickle
from glob import glob

import numpy as np
from scipy.signal import welch


def load_subject_dat(path):
    # DEAP python preprocessed files (.dat) are pickled objects.
    with open(path, "rb") as f:
        try:
            obj = pickle.load(f, encoding='latin1')
        except TypeError:
            f.seek(0)
            obj = pickle.load(f)
    return obj


def bandpower(psd, freqs, fmin, fmax):
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])


def extract_features_for_trial(eeg, sf=128):
    # eeg: channels x samples
    bands = [(1,4),(4,8),(8,13),(13,30),(30,45)]
    feats = []
    for ch in range(eeg.shape[0]):
        nperseg = min(256, eeg.shape[1])
        f, pxx = welch(eeg[ch], fs=sf, nperseg=nperseg)
        for (lo,hi) in bands:
            bp = bandpower(pxx, f, lo, hi)
            feats.append(bp)
        # basic stats per channel
        feats.append(np.mean(eeg[ch]))
        feats.append(np.std(eeg[ch]))
    return np.array(feats)


def process_deap_folder(folder, out=np.nan):
    paths = sorted(glob(os.path.join(folder, "**", "*.dat"), recursive=True))
    if not paths:
        raise FileNotFoundError("No .dat files found in %s" % folder)
    X_list = []
    y_list = []
    for p in paths:
        print("Loading", p)
        obj = load_subject_dat(p)
        # heuristics: obj may be a dict or tuple
        data = None
        labels = None
        if isinstance(obj, dict):
            data = obj.get('data')
            if data is None:
                data = obj.get('signals')
            if data is None:
                data = obj.get('X')
            labels = obj.get('labels')
            if labels is None:
                labels = obj.get('y')
        elif isinstance(obj, (list, tuple)):
            # common pattern: (data, labels)
            if len(obj) >= 2:
                data = obj[0]
                labels = obj[1]
        if data is None:
            # try attribute-like
            try:
                data = obj['data']
            except Exception:
                raise ValueError("Couldn't find data in %s" % p)

        # data expected: trials x channels x samples
        data = np.asarray(data)
        for trial_idx in range(data.shape[0]):
            eeg = data[trial_idx]
            feats = extract_features_for_trial(eeg)
            X_list.append(feats)

        # labels: many versions store ratings in `labels` or `obj[2]`
        if labels is None:
            # try to extract from obj[2] if exists
            if isinstance(obj, (list, tuple)) and len(obj) >= 3:
                labels = obj[2]
        if labels is not None:
            # labels shape: trials x 4 (valence, arousal, dominance, liking)
            # we take arousal (index 1), threshold at 5 (scale 1-9)
            labels = np.asarray(labels)
            arousal = labels[:, 1]
            y = (arousal >= 5).astype(int)
            y_list.extend(y.tolist())
        else:
            # if no labels, skip labeling; fill with zeros placeholder
            y_list.extend([0]*data.shape[0])

    X = np.vstack(X_list)
    y = np.array(y_list[:X.shape[0]])
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="DEAP folder containing .dat files")
    p.add_argument("--out", default="data/features.npz")
    args = p.parse_args()

    X, y = process_deap_folder(args.input)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    print("Saved features to", args.out)


if __name__ == "__main__":
    main()
