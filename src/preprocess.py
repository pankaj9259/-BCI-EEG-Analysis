"""Download and extract the DEAP preprocessed Python dataset.

Usage:
    python src/preprocess.py --download --out data/DEAP
"""
import argparse
import os
import zipfile

import requests

DEAP_PY_URLS = [
    "https://www.eecs.qmul.ac.uk/mmv/datasets/deap/data_preprocessed_python.zip",
    "http://www.eecs.qmul.ac.uk/mmv/datasets/deap/data_preprocessed_python.zip",
]


def download(url, outpath, chunk_size=8192):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(outpath, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)


def extract(zip_path, dest):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    p.add_argument("--out", default="data/DEAP")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    zip_path = os.path.join(args.out, "deap_python.zip")

    if args.download:
        print("Downloading DEAP preprocessed Python dataset...")
        last_error = None
        for url in DEAP_PY_URLS:
            try:
                print("Trying", url)
                download(url, zip_path)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print("Download failed from", url, "->", exc)
        if last_error is not None:
            raise last_error
        print("Extracting...")
        extract(zip_path, args.out)
        print("Done. Extracted to:", args.out)
    else:
        print("No action. Use --download to fetch dataset.")


if __name__ == "__main__":
    main()
