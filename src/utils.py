from lib2to3.pgen2 import grammar

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer


class T5Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def T5collate_fn(self, data):
    """This function collates data_list into batch.
    It is picked out from dataloader when training."""
    # encode the inputs
    task_prefix = "feedback comment: "
    encoding = tokenizer(
        [task_prefix + sequence for sequence in data["source"]],
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )  # type: BatchEncoding
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    # encode the targets
    target_encoding = tokenizer(
        data["target"], padding="longest", max_length=max_target_length, truncation=True
    )  # type: BatchEncoding
    labels = target_encoding.input_ids
    # replace padding token id's of the labels by -100
    labels = torch.tensor(labels)
    labels[labels == tokenizer.pad_token_id] = -100

    return input_ids, attention_mask, labels


class MLDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index, :]  # index means row number
        return {"text": row["text"], "labels": torch.tensor(row["labels"]).float()}

    def __len__(self):
        return len(self.df)


def MLcollate_fn(batch):
    texts, labels_list = list()
    return


def extract_grammar_terms(target):
    terms = list()

    t = target
    t = t.replace("<<", "")
    t = t.replace(">>", "")

    word = ""
    flag = False
    for i, char in enumerate(t):
        if char == ">":
            terms.append(f"<{word}>")
            word = ""
            flag = False
        elif flag:
            word += char
        elif char == "<":
            flag = True

    return sorted(list(set(terms)))


def add_info_to_source(
    source,
    offset=None,
    grammar_terms=None,
    insert_offset_bracket_into_source=False,
    offset_phrase=False,
    offset_number=False,
    add_prefix=True,
    correct_words=None,
):
    s, e = map(int, offset.split(":"))
    offset_phrase_words = source[s:e]
    if insert_offset_bracket_into_source:
        source = add_offset_info_to_source(source, offset)
    if grammar_terms is not None:
        source = add_grammar_term_info_to_source(source, grammar_terms)
    if offset_phrase:
        source = add_offset_phrase_info_to_source(source, offset_phrase_words)
    if offset_number:
        source = add_offset_number_info_to_source(source, offset)
    if correct_words is not None:
        source = " ".join([source, "<correct words>:", correct_words])
    if add_prefix:
        source = "fbc: " + source
    return source


def add_offset_info_to_source(source, offset):
    s, e = map(int, offset.split(":"))
    return source[:s] + "[ " + source[s:e] + " ]" + source[e:]


def add_offset_phrase_info_to_source(source, offset_phrase):
    source = " ".join([source, "<offset phrase>:", offset_phrase])
    return source


def add_offset_number_info_to_source(source, offset):
    s, e = offset.split(":")
    source = " ".join([source, "<offset number>:", s, e])
    return source


def add_grammar_term_info_to_source(source, grammar_terms):
    source = " ".join([source, "<grammar terms>:"] + grammar_terms)
    return source


def clean_up_data(target):
    """<>内小文字化"""
    target = target.replace("<<", "[LEFT_SPECIAL]")
    target = target.replace(">>", "[RIGHT_SPECIAL]")

    flag = False
    for i, char in enumerate(target):
        if char == ">":
            flag = False
        elif flag:
            target = target[:i] + char.lower() + target[i + 1 :]
        elif char == "<":
            flag = True
    target = target.replace("[LEFT_SPECIAL]", "<<")
    target = target.replace("[RIGHT_SPECIAL]", ">>")
    return target


def restore_data_format(target):
    target = target.replace("<<", "")
    target = target.replace(">>", "")

    new_target = []
    for sent in target.split(". "):
        if sent[0] == "<":
            if sent[1] != "<":
                sent[1] = sent[1].upper()
        new_target.append(sent)

    new_target = ". ".join(new_target)
    return new_target


def read_pred_grammar_terms(path):
    pred_grammar_terms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pred_grammar_terms.append(line.strip().split("\t"))

    return pred_grammar_terms


def dataset_to_df(dataset_path, grammar_term2id, columns, offset_phrase=False):
    data_list = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            source, offset, target = line.strip().split("\t")
            target = clean_up_data(target)
            grammar_terms = extract_grammar_terms(target)
            grammar_terms_binary = [0] * len(grammar_term2id)
            for term in grammar_terms:
                grammar_terms_binary[grammar_term2id[term]] = 1
            source = add_info_to_source(
                source,
                offset=offset,
                offset_phrase=offset_phrase,
                add_prefix=False,
            )
            new_data = ["id", source.lower()] + grammar_terms_binary
            data_list.append(new_data)
    df = pd.DataFrame(data_list, columns=columns)
    df["labels"] = df[df.columns[2:]].values.tolist()
    new_df = df[["text", "labels"]].copy()
    return new_df


def exclude_grammar_terms_by_count(
    grammar_terms: list, grammar_term2number: dict, number: int
):
    new_grammar_terms = []
    for grammar_term in grammar_terms:
        if grammar_term not in grammar_term2number:
            # print("WARNING: grammar_term not in grammar_term2number")
            continue
        if grammar_term2number[grammar_term] > number:
            new_grammar_terms.append(grammar_term)
    return new_grammar_terms


def read_grammar_term_set_count(path):
    grammar_term2number = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            grammar_term, number = line.strip().split("\t")
            number = int(number)
            grammar_term2number[grammar_term] = number
    return grammar_term2number


def read_label_names(path, ignore_labels_in_specified_line_number=-1):
    grammar_term2id = {}
    cnt = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < ignore_labels_in_specified_line_number:
                continue
            grammar_term2id[line.strip()] = cnt
            cnt += 1
    return grammar_term2id


def read_label(path):
    gts = []
    cnt = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            gts.append(line.strip())
    gts = gts[::-1]
    return gts


def read_dataset(path):
    """read dataset"""
    with open(path, encoding="utf-8") as file:
        for line in file:
            raw_src, char_based_offset, raw_tgt = line.strip().split("\t")
            char_based_offset = list(map(int, char_based_offset.strip().split(":")))
            yield raw_src, char_based_offset, raw_tgt


def make_token_based_offset(src, char_based_offset):
    start = src[: char_based_offset[0]].count(" ")
    end = src[: char_based_offset[1]].count(" ") + 1
    token_based_offset = [start, end]
    return token_based_offset


def to_char_based_offset_from_token_based_offset(src, token_based_offset):
    src = src.strip().split(" ")
    start_offset, end_offset = map(int, token_based_offset.split(" "))
    n_chars_to_start_offset = (
        sum([len(token) for token in src[:start_offset]]) + start_offset
    )
    n_chars_to_end_offset = (
        sum([len(token) for token in src[: end_offset + 1]]) + end_offset
    )
    return str(n_chars_to_start_offset) + ":" + str(n_chars_to_end_offset)


def clean_source_for_en_to_jp(src):
    new_src = src.split(" ")
    new_src = new_src[1:-1]
    return " ".join(new_src)


def prep(row, args, grammar_term2number, pred_grammar_terms, index):
    row["target_text"] = clean_up_data(row["target_text"])
    # row["source_text"] = clean_source_for_en_to_jp(row["source_text"])
    grammar_terms = None
    if args.given_grammar_terms:
        grammar_terms = extract_grammar_terms(row["target_text"])
        grammar_terms = exclude_grammar_terms_by_count(
            grammar_terms, grammar_term2number, args.exclude_grammar_terms_by_count
        )
    if (
        hasattr(args, "given_pred_grammar_terms_dev")
        and args.given_pred_grammar_terms_dev is not None
    ):
        grammar_terms = []
        if index < len(pred_grammar_terms):
            grammar_terms = pred_grammar_terms[index]
        else:
            print("WARNING: pred_grammar_terms_dev is out of range!")
    if args.target_to_grammar_terms:
        grammar_terms_for_target_text = extract_grammar_terms(row["target_text"])
        grammar_terms_for_target_text = exclude_grammar_terms_by_count(
            grammar_terms_for_target_text,
            grammar_term2number,
            args.exclude_grammar_terms_by_count,
        )
        row["target_text"] = " ".join(grammar_terms_for_target_text)
    if args.to_char_based_offset_from_token_based_offset:
        row["offset"] = to_char_based_offset_from_token_based_offset(
            row["source_text"], row["offset"]
        )
    row["source_text"] = add_info_to_source(
        row["source_text"],
        row["offset"],
        grammar_terms,
        args.insert_offset_bracket_into_source,
        args.given_offset_phrase,
        args.given_offset_number,
        correct_words=row["correct"] if args.use_crr else None,
    )

    return row
