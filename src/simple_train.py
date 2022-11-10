import argparse
import os
import pickle
import sys
from lib2to3.pgen2 import grammar

import pandas as pd
import pytorch_lightning as plain

from simplet5 import SimpleT5
import simple_predict
import utils


def init_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--train_set",
        default="data/train_dev/TRAIN.prep_feedback_comment.public.tsv",
        help="The path to train data",
    )
    parser.add_argument(
        "--dev_set",
        default="data/train_dev/DEV.prep_feedback_comment.public.tsv",
        help="The path to dev data",
    )
    parser.add_argument("--model", default="t5-small", help="The t5 model name to use")
    parser.add_argument("--out_path", required=True, help="The path for output")
    parser.add_argument(
        "--grammar_term_set",
        default="data/grammar_terms/grammar_terms.small.set",
        help="The path to grammar term data(A plain text which is grammar term for each line)",
    )
    parser.add_argument(
        "--grammar_term_set_count", default="data/grammar_terms/grammar_terms.small.set.count"
    )
    parser.add_argument(
        "--exclude_grammar_terms_by_count",
        type=int,
        default=0,
        help="Given grammar terms which are lower or equal to number set here are excluded.",
    )
    parser.add_argument(
        "--given_pred_grammar_terms_train", help="Path to pred grammar terms"
    )
    parser.add_argument(
        "--given_pred_grammar_terms_dev", help="Path to pred grammar terms"
    )
    parser.add_argument(
        "--given_grammar_terms",
        action="store_true",
        help="Flag to give grammer terms to source.",
    )
    parser.add_argument(
        "--insert_offset_bracket_into_source",
        action="store_true",
        help="Flag to insert offset bracket into source.",
    )
    parser.add_argument(
        "--given_offset_phrase",
        action="store_true",
        help="Flag to give offset phrase to source.",
    )
    parser.add_argument(
        "--given_offset_number",
        action="store_true",
        help="Flag to give offset number to source.",
    )
    parser.add_argument(
        "--target_to_grammar_terms",
        action="store_true",
        help="Flag to translate target to only grammar terms",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_preposition_tag", action="store_true")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument(
        "--to_char_based_offset_from_token_based_offset", action="store_true"
    )
    parser.add_argument("--clean_data_for_en_to_jp", action="store_true")
    parser.add_argument("--use_crr", action="store_true")
    args = parser.parse_args()

    assert (
        args.insert_offset_bracket_into_source
        or args.given_offset_phrase
        or args.given_offset_number
    )
    # assert (
    #     args.given_pred_grammar_terms_train is not None
    #     and args.given_pred_grammar_terms_dev is not None
    # )
    return args


def main():
    args = init_args()
    plain.seed_everything(args.seed)

    model = SimpleT5()

    base_model_name = 't5'
    if 'mt5' in args.model:
        base_model_name = 'mt5'
    model.from_pretrained(base_model_name, args.model)

    model.tokenizer.add_tokens(["`", "``", "''", "<<", ">>", ">", "<"])
    if args.given_grammar_terms:
        model.tokenizer.add_tokens(["<grammar terms>"])
    if args.given_offset_phrase:
        model.tokenizer.add_tokens(["<offset phrase>"])
    if args.given_offset_number:
        model.tokenizer.add_tokens(["<offset number>"])
    if args.given_offset_number:
        model.tokenizer.add_tokens(["<correct words>"])

    grammar_term2number = utils.read_grammar_term_set_count(args.grammar_term_set_count)
    if args.exclude_preposition_tag:
        grammar_term2number.pop("<preposition>")
    model.model.resize_token_embeddings(len(model.tokenizer))

    train_df = pd.read_csv(
        args.train_set,
        delimiter="\t",
        names=["source_text", "offset", "target_text"],
    )
    dev_df = pd.read_csv(
        args.dev_set,
        delimiter="\t",
        names=["source_text", "offset", "target_text"],
    )
    

    pred_grammar_terms_train = None
    if args.given_pred_grammar_terms_train is not None:
        pred_grammar_terms_train = utils.read_pred_grammar_terms(
            args.given_pred_grammar_terms_train
        )

    for index, row in train_df.iterrows():
        ret = utils.prep(
            row, args, grammar_term2number, pred_grammar_terms_train, index
        )
        train_df.loc[index, "source_text"] = ret["source_text"]
        train_df.loc[index, "target_text"] = ret["target_text"]

    pred_grammar_terms_dev = None
    if args.given_pred_grammar_terms_dev is not None:
        pred_grammar_terms_dev = utils.read_pred_grammar_terms(
            args.given_pred_grammar_terms_dev
        )
    for index, row in dev_df.iterrows():
        ret = utils.prep(row, args, grammar_term2number, pred_grammar_terms_dev, index)
        train_df.loc[index, "source_text"] = ret["source_text"]
        train_df.loc[index, "target_text"] = ret["target_text"]

    train_df = train_df[["source_text", "target_text"]]
    dev_df = dev_df[["source_text", "target_text"]]
    
    # train
    model.train(
        train_df=train_df,  # pandas dataframe with 2 columns: source_text & target_text
        eval_df=dev_df,  # pandas dataframe with 2 columns: source_text & target_text
        source_max_token_len=512,
        target_max_token_len=128,
        batch_size=8,
        max_epochs=args.epoch,
        use_gpu=True,
        outputdir=args.out_path,
        precision=32,
        save_only_last_epoch=True,
        early_stopping_patience_epochs=0,
    )


if __name__ == "__main__":
    main()
