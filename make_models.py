import subprocess
import os

for dir in os.listdir("models/triton"):
    print("Making model in {}".format(os.path.join("models/triton", dir)))
    subprocess.call(
        "mkdir 1",
        cwd=os.path.join("models/triton", dir),
        shell=True
    )
    subprocess.call(
        "python3 make_model.py",
        cwd=os.path.join("models/triton", dir),
        shell=True
    )
