import argparse
from string import whitespace

import sys
from simplet5 import SimpleT5
import pandas as pd

import utils


def init_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--test_set",
        default="data/train_dev/TEST.prep_feedback_comment.public.tsv",
        help="The path to test data",
    )
    parser.add_argument("--model_path", required=True, help="The path to trained model")
    parser.add_argument("--out_path", required=True, help="The path for output")
    parser.add_argument(
        "--given_grammar_terms",
        action="store_true",
        help="Whether you give grammar terms to input",
    )
    parser.add_argument(
        "--insert_offset_bracket_into_source",
        action="store_true",
        help="Flag to insert offset bracket into source.",
    )
    parser.add_argument(
        "--grammar_term_set_count",
        default="data/grammar_terms/grammar_terms.small.set.count",
    )
    parser.add_argument("--given_pred_grammar_terms", help="Path to pred grammar terms")
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
        help="Flag to translate target to grammar terms",
    )
    parser.add_argument(
        "--exclude_grammar_terms_by_count",
        type=int,
        default=0,
        help="Given grammar terms which are lower or equal to number set here are excluded.",
    )
    parser.add_argument("--exclude_preposition_tag", action="store_true")
    args = parser.parse_args()
    assert (
        args.insert_offset_bracket_into_source
        or args.given_offset_phrase
        or args.given_offset_number
    )
    return args


args = init_args()

print("WARNING: Don't you forget to set flag such as '--given_offset_phrase'")
grammar_term2number = utils.read_grammar_term_set_count(args.grammar_term_set_count)
if args.exclude_preposition_tag:
    grammar_term2number.pop("<preposition>")

# instantiate
model = SimpleT5()

# load trained T5 model
model.load_model("t5", args.model_path, use_gpu=True)

if args.given_pred_grammar_terms is not None:
    pred_grammar_terms = utils.read_pred_grammar_terms(args.given_pred_grammar_terms)

nlines = []
with open(args.test_set) as f:
    for index, line in enumerate(f):
        source, offset = line.strip().split("\t")
        origin_source = source
        origin_offset = offset
        grammar_terms = []
        if args.given_pred_grammar_terms is not None:
            if index < len(pred_grammar_terms):
                grammar_terms = pred_grammar_terms[index]
            else:
                print("WARNING: pred_grammar_terms_dev is out of range!")
        source = utils.add_info_to_source(
            source,
            offset,
            grammar_terms,
            args.insert_offset_bracket_into_source,
            args.given_offset_phrase,
            args.given_offset_number,
        )
        res = model.predict(source)
        res = res[0]
        res = res.replace("<< ", " <<")
        res = res.replace("'' ", " ''")
        res = res.replace("`` ", " ``")
        res = res.replace("< ", "<")
        res = res.replace("...", " ...")
        res = res.replace("` ", " `")
        white_spaces = []
        for i, char in enumerate(res):
            if i + 1 != len(res) - 1 and i != 0:
                if char == "<" and res[i + 1] != "<" and res[i - 1] != "<":
                    white_spaces.append(i)

        for s in reversed(white_spaces):
            res = res[:s] + " " + res[s:]
        res = res.strip()
        print("Source:", source)
        print("Predict:", res)
        print("")
        nlines.append(origin_source + "\t" + origin_offset + "\t" + res + "\n")

with open(args.out_path, "w") as f:
    f.writelines(nlines)
