"""
Driver that runs ON a Colab VM (via `colab exec -f scripts/colab_pipeline.py`).

It assumes the bundle produced locally (scripts/ + prepared/) was uploaded and
extracted to /content/proj on the VM. It trains BOTH models on the already-prepared
data, evaluates them, scores both at high SNR, then tars the outputs so you can pull
everything back with a single `colab download /content/out.tar`.

Nothing here touches the 19.5 GB dataset — data_prep already ran on the laptop.
"""

import subprocess
import sys

ROOT = "/content/proj"          # where the uploaded bundle was extracted on the VM


def run(*args):
    print(f"\n>>> {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


for model in ("vgg", "resnet"):
    run(f"{ROOT}/scripts/train.py", "--model", model)
    run(f"{ROOT}/scripts/evaluate.py", "--model", model)

run(f"{ROOT}/scripts/high_snr.py")   # scores both models side by side

# Bundle the trained models + figures into one file for a single download.
subprocess.run(["tar", "cf", "/content/out.tar", "-C", ROOT, "results", "models"],
               check=True)
print("\nDONE — pull results with:  colab download /content/out.tar ./colab_out.tar")
