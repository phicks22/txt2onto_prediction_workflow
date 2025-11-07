from pathlib import Path
from scipy.sparse import csr_matrix
import json
import re
import nltk
import string
import numpy as np
import pandas as pd
import numpy.typing as npt
import polars as pl
import sys
import pickle


def check_outdir(_dir: str) -> Path:
    outdir = Path(_dir)
    if not outdir.exists():
        outdir.mkdir(exist_ok=True, parents=True)
    return outdir


def calc_cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2.T) / np.dot(
        np.linalg.norm(emb1, axis=1)[:, None], np.linalg.norm(emb2, axis=1)[None, :]
    )


def get_sample_inds(folds: dict, id: np.array) -> list[list[tuple[list, list]]]:
    train_test_inds = []
    for i in sorted(folds.keys()):
        train = folds[i]["train"]
        test = folds[i]["test"]
        train_inds = np.array(
            [np.argwhere(id == sample).reshape(-1)[0] for sample in train]
        )
        test_inds = np.array(
            [np.argwhere(id == sample).reshape(-1)[0] for sample in test]
        )
        train_test_inds.append([train_inds, test_inds])
    return train_test_inds


def load_fold(fold_file: str, task: str, mode: str) -> dict:
    task_ = task.replace(":", "_")
    with open(f"{fold_file}/{task_}.json", "r") as f:
        folds = json.loads(f.read())
    if len(folds) > 0:
        if mode == "cv":
            if "CV" in list(folds.keys()):
                return folds["CV"]
            else:
                print("Not enough folds for CV")
                exit(0)

        elif mode == "simple_train":
            return folds

        elif mode == "test":
            # get training index
            train_inds = []
            for i in sorted(folds.keys()):
                train_inds.extend(folds[i]["train"])
                train_inds.extend(folds[i]["test"])
            # get testing index
            test_inds = folds["test"]

            # synthesize folds
            test_fold = {}
            test_fold.setdefault(0, {})
            test_fold[0]["train"] = sorted(list(set(train_inds)))
            test_fold[0]["test"] = test_inds

            return test_fold
        elif mode == "full_test":
            test_fold = {}
            test_fold.setdefault(0, {})
            test_fold[0]["train"] = folds["train"]
            test_fold[0]["test"] = folds["test"]

            return test_fold
    else:
        return {}


def load_sample(id: str) -> np.array:
    return np.load(id, allow_pickle=True)["gsms"]


def load_label(task: str, label: str) -> pl.DataFrame:
    label_data = pl.read_parquet(label, columns=["index", task])
    _index = label_data.drop_in_place("index").to_numpy().flatten()

    if label_data.is_empty():
        sys.exit(f"Column {task} not in {label}.")

    return label_data.to_numpy().flatten(), _index


def load_corpus(file_: str) -> tuple[list, list, list]:
    """
    Load corpus data from .npz file.

    :param file_:  /path/to/corpus.npz storing the corpus and an index for each document.
    """
    data = np.load(file_, allow_pickle=True)
    return data["corpus"], data["index"]


def load_expression(
    file: str, columns: list[str] | str = "all"
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.str_], npt.NDArray[np.str_]]:
    """
    Creates an expression object from a parquet file. The file must have
    an expression matrix of shape (genes, index) with columns of index IDs.
    There must also be a column named "genes" outlining gene IDs for each row.

    Args
    ----
    file: (FilePath)
        Path to expression.parquet.
    columns: (list[str] | str)
        Either all columns or a list of column IDs to load.
    """

    # Load the parquet file
    if columns == "all":
        df = pl.read_parquet(file)
    if isinstance(columns, list):
        columns.append("gene")
        df = pl.read_parquet(file, columns=columns)
    else:
        raise ValueError(
            f"Expected columns as type list[str] | str, got {type(columns)}."
        )

    # Create the Expression object
    genes = df.drop_in_place("gene").to_numpy().reshape(-1)
    index = np.array(df.columns)
    return df.to_numpy().T, genes, index


def load_ner_data(corpus: str) -> np.array:
    data = []
    with open(corpus) as f:
        for line in f:
            data.append(line.strip().split("||"))
    return data


def load_embeddings(file: str) -> tuple[list, list]:
    file_ = np.load(file)
    return file_["embeddings"], file_["words"]


def word_to_inds(words: np.array, word_features: np.array) -> np.array:
    return np.array([np.argwhere(words == i).reshape(-1)[0] for i in word_features])


def save_output(perf: list, outdir: Path, task: str):
    task_ = task.replace(":", "_")
    out = pd.DataFrame(
        perf,
        columns=[
            "fold",
            "auprc",
            "f1",
            "balanced_accuracy",
            "train_pos",
            "train_neg",
            "test_pos",
            "test_neg",
            "control_train_labels",
            "control_test_labels",
            "c",
            "l1r",
        ],
    )
    out.to_csv(
        f"{outdir}/{task_}.csv",
        index=False,
    )


def save_beta(betas, outdir, task):
    task_ = task.replace(":", "_")
    for i in betas.keys():
        np.savez_compressed(
            f"{outdir}/{task_}_{i}.npz",
            betas=betas[i]["coef"],
            features=betas[i]["beta"],
        )


def save_beta_pkl(betas, outdir, task):
    task_ = task.replace(":", "_")
    for i in betas.keys():
        param = {
            "coef": betas[i]["coef"],
            "model": betas[i]["model"],
            "beta": betas[i]["beta"],
            "prior": betas[i]["prior"],
        }
        with open(f"{outdir}/{task_}__model_{i}.pkl", "wb") as f:
            pickle.dump(param, f)


def remove_unencoded_text(text):
    """
    Removes characters that are not UTF-8 encodable.
    """
    return "".join([i if ord(i) < 128 else "" for i in text])


def is_allowed_word(word, stopwords, remove_numbers, min_word_len):
    """
    Checks if word is allowed based on inclusion in stopwords, presence of
    numbers, and length.
    """
    stopwords_allowed = word not in stopwords
    numbers_allowed = not (remove_numbers and contains_numbers(word))
    length_allowed = len(word) >= min_word_len
    return stopwords_allowed and numbers_allowed and length_allowed


def contains_numbers(text):
    """
    Parses text using a regular expression and returns a boolean value
    designating whether that string contains any numbers.
    """
    return bool(re.search(r"\d", text))


def sparse_mat_mul(a, b):
    sparse_matrix_a = csr_matrix(a)
    sparse_matrix_b = csr_matrix(b)
    result_sparse = sparse_matrix_a.dot(sparse_matrix_b)
    return result_sparse.toarray()


def underscore_to_colon(w1):
    if "_" in w1:
        w2 = re.sub("_", ":", w1)
    else:
        w2 = w1
    return w2
