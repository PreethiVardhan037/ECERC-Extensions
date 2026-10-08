"""
[ADDED] Fetch the authors' preprocessed features from their public Google Drive folder and place them
where the code expects them (ECERC_fixed/data/meld and ECERC_fixed/data/iemocap).

    pip install gdown
    python download_data.py                    # download + arrange + verify
    python download_data.py --from_dir PATH    # skip download: arrange files you already downloaded (e.g. a Kaggle input dir)

Files are found by name anywhere inside the downloaded tree, so the Drive folder layout does not matter.
"""
import argparse, os, shutil, subprocess, sys

DRIVE = "https://drive.google.com/drive/folders/1DGoPEBvMgMOge7u20wjKn6slYe4awyQi"
HERE = os.path.dirname(os.path.abspath(__file__))
NEED = {
    "meld": ["meld_emotion_semantic_features_roberta.pkl"],
    "iemocap": ["IEMOCAP_features.pkl",
                "iemocap_emotion_features_roberta.pkl",
                "iemocap_emotion_semantic_features_roberta.pkl"],
}


def arrange(src_root):
    index = {}
    for dp, _, fs in os.walk(src_root):
        for f in fs:
            index.setdefault(f, os.path.join(dp, f))
    ok = True
    for ds, files in NEED.items():
        dst_dir = os.path.join(HERE, "data", ds)
        os.makedirs(dst_dir, exist_ok=True)
        for f in files:
            dst = os.path.join(dst_dir, f)
            if os.path.exists(dst):
                print("[ok]      ", dst); continue
            if f in index:
                shutil.copy2(index[f], dst); print("[copied]  ", dst)
            else:
                print("[MISSING] ", f, "-> not found under", src_root); ok = False
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_dir", default=None, help="folder that already contains the downloaded .pkl files")
    a = ap.parse_args()
    src = a.from_dir
    if src is None:
        src = os.path.join(HERE, "data_download")
        subprocess.check_call([sys.executable, "-m", "gdown", "--folder", DRIVE, "-O", src])
    if arrange(src):
        print("\nAll dataset files are in place. You can now run train.py / inference.py / calibrate.py.")
    else:
        print("\nSome files are missing. If gdown hit Google's download quota, download the Drive folder in your "
              "browser and re-run:  python download_data.py --from_dir <that folder>")
        sys.exit(1)
