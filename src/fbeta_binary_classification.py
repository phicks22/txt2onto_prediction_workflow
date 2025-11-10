"""
This script applies the fbeta-score to best classify the predicted probabilities to
binary classes.

Authors: Junxia Lin, Parker Hicks
Date: 2025-07-02
Last update: 2025-10-07
"""

import polars as pl
from glob import glob
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.metrics import fbeta_score
from sklearn.metrics import average_precision_score as auPRC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import StratifiedKFold
import tracemalloc


def load_system_descendants(constraints: str | Path | None) -> np.ndarray | None:
    """
    Load system descendants from a JSON file and return a unique list of all descendant IDs.

    Parameters:
    - constraints: Path to the JSON file containing system descendants, or None.

    Returns:
    - A NumPy array of unique descendant identifiers, or None if no constraints are provided.
    """
    
    if not constraints:
        return None

    constraints_path = Path(constraints)
    if not constraints_path.exists():
        raise FileNotFoundError(f"Constraints file not found: {constraints_path}")

    with open(constraints_path, "r") as f:
        system_descendants = json.load(f)

    sys_desc = []
    for key in system_descendants:
        sys_desc.extend(system_descendants[key])

    return np.unique(np.array(sys_desc))


def find_best_threshold_fbeta(y_true, y_probs, beta):
    best_threshold = 0.0
    best_fbeta = 0.0

    # Use sorted unique predicted probabilities as candidate thresholds
    thresholds = np.sort(np.unique(y_probs))[::-1]

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).cast(pl.Int8)
        score = fbeta_score(y_true.to_numpy(), y_pred.to_numpy(), beta=beta)
        if score > best_fbeta:
            best_fbeta = score
            best_threshold = threshold

    return best_threshold, best_fbeta


def stratified_split(X, y, groups=None, min_pos=3, n_splits=5, random_state=42):
    """
    Perform a stratified group split with additional constraints:
        1. Stratified: preserves label distribution.
        2. (for sample) Grouped: ensures no group appears in both train and test sets.
        3. Test set must contain more than min_pos positive samples/study (convering positives>0).

    Returns:
        (train_set, test_set): Lists of training and testing samples/study.

    Returns (None, None) if no valid split is found.

    """

    # to have final 8/2 split ratio, use n_splits=5 for the number of splits
    # sample
    if groups is not None:
        print("doing StratifiedGroupKFold with group.")
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        for train_idx, test_idx in sgkf.split(X, y, groups):
            tst_pos = np.sum(y[test_idx] == 1)
            conditions = [tst_pos >= min_pos]
            if np.all(conditions):
                return X[train_idx].tolist(), X[test_idx].tolist()
    # study
    else:
        print("doing StratifiedKFold without group.")
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        for train_idx, test_idx in skf.split(X, y):
            tst_pos = np.sum(y[test_idx] == 1)
            conditions = [tst_pos >= min_pos]
            if np.all(conditions):
                return X[train_idx].tolist(), X[test_idx].tolist()

    return None, None


def best_threshold_classify(
    pred_prob: pl.DataFrame,
    train: pl.DataFrame,
    test: pl.DataFrame,
    task: str,
    beta: float,
    ) -> pl.DataFrame:
    """Function to calculate the best threshold using 20% of the data 
    and apply it to the rest 80% data. Also apply the best threshold to
    the original full predicted probability data."""

    # run fbeta-score to get the best threshold using the train (20%) set. 
    ground_truths = train[task]
    predicted_values = train["prob"]
    best_threshold, best_fbeta = find_best_threshold_fbeta(
        ground_truths, predicted_values, beta
    )

    # apply the best_threshold to the test (80%) set.
    train_bin = test.with_columns(
        (pl.col("prob") > best_threshold).cast(pl.Int8).alias("pred_binary")
    )

    # apply the best threshold to the original full predicted probabilities
    pred_prob_bin = pred_prob.with_columns(
        (pl.col("prob") > best_threshold).cast(pl.Int8).alias("pred_binary")
    )

    return pred_prob_bin, train_bin, best_threshold


def save_output(fold: dict, outdir: str, task_name: str) -> None:
    """Save folds to json for specified task name."""
    # Set folds directory

    outdir = os.path.join(outdir, "training_index_pred_annotation")
    os.makedirs(outdir, exist_ok=True)
    outdir_path = Path(outdir)
    save_task = task_name.replace(":", "_")  # To make machine readible

    # Save folds dict as json
    if len(fold) > 0:
        with open(outdir_path / f"{save_task}.json", "w") as f:
            f.write(json.dumps(fold, indent=4))
        print(f"Training set saved for {task_name}")


def fbeta_binary_classification(
    label_dir: str,
    prob_dir: str,
    index_out_dir: str,
    outdir: str,
    beta: float,
    constraints: str,
    is_study: bool,
) -> None:
    """Function to apply the fbeta method to classify the predicted probabilities
    into binary classes.
    """

    pred_prob_data_agg_bin = None
    label_data = pl.scan_parquet(label_dir)
    columns_to_select = [] if is_study else ["group"]
    selected_columns = columns_to_select + ["index"]
    pred_label_agg_data = label_data.select(selected_columns).collect()

    # filter for the system descendants
    descendants = load_system_descendants(constraints)

    # loop over the prediction file of each term
    all_best_th = []
    for file in tqdm(glob(f"{prob_dir}/*.csv"), total=len(glob(f"{prob_dir}/*.csv"))):
        task = Path(file).stem.removesuffix("__preds").replace("_", ":")
        if descendants is not None and task in descendants:
            
            # trace memory
            tracemalloc.start(15)
            
            print(f"Generating binary classification for {task}")

            # read in the predicted probability data
            pred_prob_data = pl.read_csv(file).select(["ID", "prob"])

            # subset the label data to only the group, index (sample/study), and the label term columns
            # Remove rows where the label is 0
            # Convert negative values to zero in the label column
            selected_columns = columns_to_select + ["index", task]
            label_data_sub = (
                label_data.select(selected_columns)
                .filter(pl.col(task) != 0)
                .with_columns(pl.col(task).clip(lower_bound=0))
                .collect()
            )

            # merge the prediction data to label data
            # this subsets the prediction data to match label data - having the 
            # same index (samples/study) with label data
            pred_label_data = pred_prob_data.join(
                label_data_sub, left_on="ID", right_on="index", how="right"
            )
            
            # data with probability and true label
            selected_columns = columns_to_select + ["index", "prob", task]
            pred_label_data = pred_label_data.select(selected_columns)

            # drop the rows with na (there could be NA in prob).
            pred_label_data = pred_label_data.drop_nulls()

            # stratified Kfold split
            X = pred_label_data["index"].to_numpy()
            y = pred_label_data[task].to_numpy()
            
            # split (5 folds by default)
            groups = None if is_study else pred_label_data[columns_to_select[0]].to_numpy()
            train_set, test_set = stratified_split(X, y, groups)

            if test_set:
                pred_label_data_tst = pred_label_data.filter(
                    pl.col("index").is_in(test_set)
                )
                pred_label_data_trn = pred_label_data.filter(
                    pl.col("index").is_in(train_set)
                )

                # fbeta binary classification
                # get the best threshold
                # apply the threshold to the full set of predicted probabilities. 
                # Note: test_set (20% of the data) from stratified_split is used 
                # as the training data to get the best threshold below.
                pred_prob_data_bin, pred_label_data_trn_bin, best_threshold = (
                    best_threshold_classify(
                        pred_prob_data,
                        pred_label_data_tst,
                        pred_label_data_trn,
                        task,
                        beta,
                    )
                )

                # record apop
                ground_truths = pred_label_data_tst[task].to_numpy()
                predicted_values = pred_label_data_tst["prob"].to_numpy()
                prior = np.mean(ground_truths)
                apop = np.log2(
                    auPRC(ground_truths, predicted_values) / prior
                )
                
                # collect the best threshold
                best_th = [task, best_threshold, prior, apop]
                all_best_th.append(best_th)

                # merge the classified results to the aggregate dataset - train data
                selected_columns = columns_to_select + ["index", "pred_binary"]
                pred_label_data_trn_bin = pred_label_data_trn_bin.select(
                                                selected_columns
                                            ).rename({"pred_binary": task})
                selected_columns = columns_to_select + ["index"]
                pred_label_agg_data = pred_label_agg_data.join(
                    pred_label_data_trn_bin, on=selected_columns, how="left"
                )

                # merge the classified results to the aggregate dataset - full predicted probability samples
                pred_prob_data_bin = pred_prob_data_bin.select(["ID", "pred_binary"]).rename({"pred_binary": task})
                
                if pred_prob_data_agg_bin is None:
                    pred_prob_data_agg_bin = pred_prob_data_bin
                else:
                    pred_prob_data_agg_bin = pred_prob_data_agg_bin.join(
                        pred_prob_data_bin, on="ID", how="left"
                    )
                
                # save the index (study/sample) id
                train_index = {}
                train_index["train"] = train_set
                save_output(train_index, index_out_dir, task)

            else:
                print(f"No valid split for {task}, skipping it.")

            # trace memory
            current, peak = tracemalloc.get_traced_memory()
            print(f"tracemalloc current={current/1e6:.1f} MB, peak={peak/1e6:.1f} MB")
            tracemalloc.stop()
    
        else:
            print(f"Either system descendants are not provided or {task} is not a system descendant.")
            
    # save the best thresholds
    best_th_df = pl.DataFrame(
        all_best_th,
        schema=["task", "best_threshold", "prior", "log2(auprc/prior)"]
    )

    best_th_df.write_csv(
        f"{index_out_dir}/f{beta}_best_threshold.csv",
        separator="\t"
    )

    # remove rows where all columns except "group" and "index" are null (NA)
    selected_columns = columns_to_select + ["index"]
    pred_label_agg_data = pred_label_agg_data.filter(
        ~pl.all_horizontal(
            [
                pl.col(c).is_null()
                for c in pred_label_agg_data.columns
                if c not in selected_columns
            ]
        )
    )

    # Fill nulls with 0
    pred_label_agg_data = pred_label_agg_data.fill_null(0)
    pred_prob_data_agg_bin = pred_prob_data_agg_bin.fill_null(0)
    # save the annotations
    pred_prob_data_agg_bin.write_parquet(outdir)


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
        "-index_out_dir",
        help="Path to save the index for following training",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-outdir", help="outdir to classified labels", required=True, type=str
    )
    parser.add_argument(
        "-beta", help="beta in the fbeta-score calculation", default=1.0, type=float
    )
    parser.add_argument(
        "-constraints",
        help="Path to the json file of the terms to limit to.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-is_study",
        help="Flag if the data is sample-level or study-level",
        default=False,
        type=bool,
    )
    args = parser.parse_args()

    fbeta_binary_classification(
        label_dir=args.label_dir,
        prob_dir=args.prob_dir,
        index_out_dir=args.index_out_dir,
        outdir=args.outdir,
        beta=args.beta,
        constraints=args.constraints,
        is_study=args.is_study,
    )
