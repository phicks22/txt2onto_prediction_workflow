"""
This script generates the prior, number of positives, and the number of predicted positives
from the ground truth labels and annotations.

Authors: Junxia Lin
Date: 2025-06-27
"""

import polars as pl
from glob import glob
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-label_dir", help="Path to the label data", required=True, type=str
    )
    parser.add_argument(
        "-prob_dir",
        help="Path to the predicted probability results",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-best_threshold",
        help="Path to the best threshold data",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-annotations",
        help="Path to the annotation data",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-is_study",
        help="Flag if the data is sample-level or study-level",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "-outdir", help="outdir to calculated data", required=True, type=str
    )
    args = parser.parse_args()

    label_data = pl.scan_parquet(args.label_dir)
    annotations = pl.scan_parquet(args.annotations)
    best_th_df = pd.read_csv(args.best_threshold, sep="\t")
    best_th_df = best_th_df[best_th_df["task"] != "task"].reset_index(drop=True)

    prior = [pd.NA] * len(best_th_df["task"])
    num_pos = [pd.NA] * len(best_th_df["task"])
    num_pred_pos = [pd.NA] * len(best_th_df["task"])

    # loop over the prediction file of each term
    for file in tqdm(
        glob(f"{args.prob_dir}/*.csv"), total=len(glob(f"{args.prob_dir}/*.csv"))
    ):
        task = Path(file).stem.removesuffix("__preds").replace("_", ":")
        if task in best_th_df["task"].to_list():
            pred_data = pl.read_csv(file)

            # subset the label data to only the group, sample, and the label term columns
            # Remove rows where the label is 0
            # Convert negative values to zero in the label column
            columns_to_select = [] if args.is_study else ["group"]
            selected_columns = columns_to_select + ["index", task]
            label_data_sub = (
                label_data.select(selected_columns)
                .filter(pl.col(task) != 0)
                .with_columns(pl.col(task).clip(lower_bound=0))
                .collect()
            )

            print("pred_data and label_data_sub")
            print(pred_data)
            print(label_data_sub)
            # merge the prediction data to label data
            pred_label_data = pred_data.select(["ID", "prob"]).join(
                label_data_sub, left_on="ID", right_on="index", how="right"
            )
            selected_columns = columns_to_select + ["index", task, "prob"]
            pred_label_data = pred_label_data.select(selected_columns)

            # drop the rows with na (there could be NA in prob).
            pred_label_data = pred_label_data.drop_nulls()
            print("pred_label_data")
            print(pred_label_data)
            
            # subset the annotaion data
            annotation_sub = annotations.select(["ID", task]).collect()
            annotation_sub = annotation_sub.rename({"ID": "index"})
            print("annotation_sub")
            print(annotation_sub)

            # prior
            prior_value = pred_label_data.select(pl.mean(task)).item()
            # number of positives
            num_pos_value = pred_label_data.select((pl.col(task) > 0).sum()).item()
            # number of predicted positives
            num_pred_pos_value = annotation_sub.select((pl.col(task) > 0).sum()).item()

            # write them in the list
            index = best_th_df[best_th_df["task"] == task].index[0]
            prior[index] = prior_value
            num_pos[index] = num_pos_value
            num_pred_pos[index] = num_pred_pos_value

    # save the data (change: use polars dataframe)
    combined_data = pd.DataFrame(
        {
            "task": best_th_df["task"],
            "best_threshold": best_th_df["best_threshold"],
            "prior": prior,
            "num_of_pos": num_pos,
            "num_of_pred_pos": num_pred_pos,
        }
    )
    combined_data.to_csv(args.outdir, index=False)
