import os
import re

import pandas as pd
import sastvd as svd
import sastvd.helpers.git as svdg
from sklearn.model_selection import train_test_split

def train_val_test_split_df(df, idcol, labelcol):
    """Add train/val/test column into dataframe."""
    X = df[idcol]
    y = df[labelcol]
    def path_to_label(path):

        return "test"

    df["label"] = df[idcol].apply(path_to_label)
    return df

def remove_comments(text):
    """Delete comments from code."""

    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)

def InputData(minimal=True, sample=False, return_raw=False, splits="default"):
    """Read Test Data.

    Args:
        sample (bool): Only used for testing!
        splits (str): default, crossproject-(linux|Chrome|Android|qemu)

    EDGE CASE FIXING:
    id = 177860 should not have comments in the before/after
    """
    savedir = svd.get_dir(svd.cache_dir() / "test_datasets")
    if minimal:
        try:
            df = pd.read_parquet(
                savedir / f"test_datasets_{sample}.pq", engine="fastparquet"
            ).dropna()
            # print(df.keys())
            # print(df.head())
            # print(df.vul.tail(100))
            md = pd.read_csv(svd.cache_dir() / "test_case/test_metadata.csv")
            # md.groupby("project").count().sort_values("id")

            default_splits = svd.external_dir() / "test_rand_splits.csv"
            if os.path.exists(default_splits):
                splits = pd.read_csv(default_splits)
                splits = splits.set_index("id").to_dict()["label"]
                df["label"] = df.id.map(splits)

            # if "crossproject" in splits:
            #     project = splits.split("_")[-1]
            #     md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
            #     nonproject = md[md.project != project].id.tolist()
            #     trid, vaid = train_test_split(nonproject, test_size=0.1, random_state=1)
            #     teid = md[md.project == project].id.tolist()
            #     teid = {k: "test" for k in teid}
            #     trid = {k: "train" for k in trid}
            #     vaid = {k: "val" for k in vaid}
            #     cross_project_splits = {**trid, **vaid, **teid}
            #     df["label"] = df.id.map(cross_project_splits)

            return df
        except Exception as E:
            print(E)
            pass
    filename = "test_case.csv"
    df = pd.read_csv(svd.external_dir() / filename)
    # print(df.keys())
    df = df.rename(columns={"Unnamed: 0": "id"})
    # print("After rename:", df.keys())

    df["dataset"] = "test_case"

    # Remove comments
    df["func_before"] = svd.dfmp(df, remove_comments, "func_before", cs=500)
    df["func_after"] = svd.dfmp(df, remove_comments, "func_after", cs=500)

    # print("Remove comments", df.shape)
    # Return raw (for testing)
    if return_raw:
        return df

    # Save codediffs
    cols = ["func_before", "func_after", "id", "dataset"]
    svd.dfmp(df, svdg._c2dhelper, columns=cols, ordr=False, cs=300)
    # print("Save codediffs:", df.shape)

    # Assign info and save
    df["info"] = svd.dfmp(df, svdg.allfunc, cs=500)
    df = pd.concat([df, pd.json_normalize(df["info"])], axis=1)

    # Make splits
    df = train_val_test_split_df(df, "id", "vul")
    # print(df["vul"].tail(100))
    # print(df.before.head())

    keepcols = [
        "dataset",
        "id",
        "label",
        "removed",
        "added",
        "diff",
        "before",
        "after",
        "vul",
    ]
    df_savedir = savedir / f"test_datasets_{sample}.pq"
    df[keepcols].to_parquet(
        df_savedir,
        object_encoding="json",
        index=0,
        compression="gzip",
        engine="fastparquet",
    )
    metadata_cols = df.columns[:17].tolist() + ["project"]
    df[metadata_cols].to_csv(svd.cache_dir() / "test_case/test_metadata.csv", index=0)
    return df
