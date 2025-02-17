import os
import pickle as pkl
import sys

import numpy as np
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.joern as svdj
import sastvd.helpers.sast as sast
from warnings import filterwarnings

# SETUP
NUM_JOBS = 100
print(sys.argv)
# 0-5 55-60 90-99
JOB_ARRAY_NUMBER = 0 if "ipykernel" in sys.argv[0] else int(sys.argv[1]) - 1

# Read Data
df = svdd.bigvul()
df = df.iloc[::-1]
splits = np.array_split(df, NUM_JOBS)
print(len(splits))

def filter_warnings():
    # "The dataloader does not have many workers which may be a bottleneck."
    filterwarnings("ignore",
                   category=FutureWarning)


def preprocess(row):
    """Parallelise svdj functions.

    Example:
    df = svdd.bigvul()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """
    # print("process", row["id"])
    savedir_before = svd.get_dir(svd.processed_dir() / row["dataset"] / "before")
    savedir_after = svd.get_dir(svd.processed_dir() / row["dataset"] / "after")

    # Write C Files
    fpath1 = savedir_before / f"{row['id']}.c"
    with open(fpath1, "w") as f:
        f.write(row["before"])
    fpath2 = savedir_after / f"{row['id']}.c"
    if len(row["diff"]) > 0:
        with open(fpath2, "w") as f:
            f.write(row["after"])

    # Run Joern on "before" code
    if not os.path.exists(f"{fpath1}.edges.json"):
        svdj.full_run_joern(fpath1, verbose=1)

    # Run Joern on "after" code
    if not os.path.exists(f"{fpath2}.edges.json") and len(row["diff"]) > 0:
        svdj.full_run_joern(fpath2, verbose=1)

    # Run SAST extraction
    fpath3 = savedir_before / f"{row['id']}.c.sast.pkl"
    if not os.path.exists(fpath3):
        sast_before = sast.run_sast(row["before"])
        with open(fpath3, "wb") as f:
            pkl.dump(sast_before, f)


if __name__ == "__main__":
    filter_warnings()
    # for job in range(61,65):
    #     svd.dfmp(splits[job], preprocess, ordr=False, workers=12, desc="Get Graph:")
