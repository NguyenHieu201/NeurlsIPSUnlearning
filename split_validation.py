import os
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
import typer


def save_df(df: pd.DataFrame, path: str):
    df.reset_index(inplace=True, drop=True)
    df.to_csv(path, index=False)


def main(num_part: int, data_path: str, seed: Optional[int] = 0):
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)

    ori_train_df, ori_val_df = train_test_split(
        train_df, test_size=0.2, random_state=seed)

    save_df(ori_train_df, "./data/train.csv")
    save_df(ori_val_df, "./data/val.csv")
    save_df(test_df, "./data/test.csv")

    for i in range(num_part):
        cross_validation_dir = f"./data/validation_{i}"
        if not os.path.exists(cross_validation_dir):
            os.makedirs(cross_validation_dir)
        retain_df, forget_df = train_test_split(
            ori_train_df, test_size=0.2, random_state=i
        )
        save_df(retain_df, f"{cross_validation_dir}/retain.csv")
        save_df(forget_df, f"{cross_validation_dir}/forget.csv")


if __name__ == "__main__":
    typer.run(main)
