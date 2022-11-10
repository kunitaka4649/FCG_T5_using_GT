import argparse
import os
import random
import sys
from lib2to3.pgen2 import grammar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from sklearn import metrics, model_selection, preprocessing
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

import utils


def seed_everything(seed=73):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


seed_everything(1234)

# --- add ---


def init_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset", default="data/train_dev/DEV.prep_feedback_comment.public.tsv"
    )
    parser.add_argument("--out_path", required=True)
    parser.add_argument(
        "--model",
        default="roberta-base",
        choices=["bert-base-uncased", "roberta-base", "roberta-large"],
    )
    parser.add_argument("--grammar_term_set", default="data/grammar_terms.small.set")
    parser.add_argument(
        "--given_offset_phrase",
        action="store_true",
        help="Flag to give offset phrase to source.",
    )
    parser.add_argument("--exclude_grammar_terms_by_count", type=int, default=0)
    parser.add_argument("--exclude_preposition_tag", action="store_true")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("-lr", "--learning_rate", type=float, default=4e-5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--class_weight_pkl", default="data/grammar_terms_weight.pkl")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--dont_save_model", action="store_true")
    parser.add_argument("--thresh", type=float, default=0)
    return parser.parse_args()


def dataset_to_df(
    dataset_path,
    grammar_term2id,
    columns,
    exclude_grammar_terms_by_count,
    offset_phrase=False,
):
    data_list = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            source, offset, target = line.strip().split("\t")
            target = utils.clean_up_data(target)
            grammar_terms = utils.extract_grammar_terms(target)
            grammar_terms = utils.exclude_grammar_terms_by_count(
                grammar_terms, grammar_term2id, 0
            )
            grammar_term_ids = []
            for term in grammar_terms:
                grammar_term_ids.append(grammar_term2id[term])
            source = utils.add_info_to_source(
                source,
                offset=offset,
                offset_phrase=offset_phrase,
                add_prefix=False,
            )
            new_data = ["id", source.lower()] + [grammar_term_ids]
            data_list.append(new_data)
    df = pd.DataFrame(data_list, columns=columns)
    return df


args = init_args()

grammar_term2id = utils.read_label_names(
    args.grammar_term_set, args.exclude_grammar_terms_by_count
)

if args.exclude_preposition_tag:
    grammar_term2id.pop("<preposition>")

columns = ["id", "text", "labels"]
train = dataset_to_df(
    args.dataset,
    grammar_term2id,
    columns,
    args.exclude_grammar_terms_by_count,
    args.given_offset_phrase,
)

id2grammar_term = {value: key for key, value in grammar_term2id.items()}

with open(args.out_path, "w", encoding="utf-8") as f:
    for i, row in train.iterrows():
        for term in row["labels"]:
            print(id2grammar_term[term], end="\t", file=f)
        print("", file=f)
