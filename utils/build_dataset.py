import pandas as pd
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from typing import Tuple
from PIL import Image

from sklearn.model_selection import GroupShuffleSplit

def split_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["labels"] = df["Finding Labels"].str.lower().str.split("|")
    return df


def split_df_by_pathology(df: pd.DataFrame, pathology: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    positive_df = df[df["labels"].apply(lambda x: pathology.lower() in x)]
    negative_df = df[df["labels"].apply(lambda x: pathology.lower() not in x and "no finding" not in x)]
    return positive_df, negative_df


def sample_df(df: pd.DataFrame, size: int):
    return df.sample(n=size, random_state=42)


def assign_targets(df: pd.DataFrame, pathology: str) -> pd.DataFrame:
    df["target"] = df["labels"].apply(
        lambda x: 1 if pathology.lower() in x else 0
    )
    return df


def split_df_for_train_test(df: pd.DataFrame, split: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if split == 0.0: 
        return df, df
    
    gss = GroupShuffleSplit(
        test_size=split,
        n_splits=1,
        random_state=42
    )
    
    train_idx, test_idx = next(
        gss.split(
            df,
            groups=df["Patient ID"]
        )
    )
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    return train_df, test_df


def summarize(df: pd.DataFrame) -> pd.Series:
    return df["target"].value_counts(normalize=True)


def class_ratio(df: pd.DataFrame) -> float:
    return df["target"].mean()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    
    out_dir = Path(cfg.data.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the metadata
    input_dir = Path(cfg.data.src_dir)
    input_df = pd.read_csv(input_dir / 'Data_Entry_2017.csv')
    
    # Split labels
    input_df = split_labels(input_df)
    
    # Split the dataframe by pathology
    pos_df, neg_df = split_df_by_pathology(input_df, cfg.data.pathology)
    
    # Sample the datasets
    pos_df = sample_df(pos_df, min(cfg.data.dataset_size//2, len(pos_df)))
    neg_df = sample_df(neg_df, cfg.data.dataset_size - len(pos_df))
    
    # Concatenate the Dataset
    subset_df = pd.concat((pos_df, neg_df))
    
    # Assign targets
    subset_df = assign_targets(subset_df, cfg.data.pathology)
    
    # Create Train, Validation and Test splits
    trainval_df, test_df = split_df_for_train_test(subset_df, cfg.data.train_test_split)
    train_df, val_df = split_df_for_train_test(trainval_df, cfg.data.train_val_split)
    
    # Verify label balance per split
    print("Train:", summarize(train_df))
    print("Val:", summarize(val_df))
    print("Test:", summarize(test_df))
    
    # Save the splits
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    
    