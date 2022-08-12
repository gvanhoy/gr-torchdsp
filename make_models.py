import subprocess
import os

for dir in os.listdir("models"):
    print("Making model in {}".format(os.path.join("models", dir)))
    subprocess.call(
        "mkdir 1",
        cwd=os.path.join("models", dir),
        shell=True
    )
    subprocess.call(
        "python3 make_model.py",
        cwd=os.path.join("models", dir),
        shell=True
    )
