"""output count grammar terms
output count grammar terms (<grammar term>\t<count>)"""
import argparse
import json
import math
import pickle

import utils


def init_args():
    """initial args"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--out", default="data/grammar_terms.small.set.count")
    parser.add_argument("--exclude_grammar_terms_by_count", type=int, default=-1)
    parser.add_argument("--only_top_n_vocab", action="store_true")
    parser.add_argument("--n_terms", type=int, default=10)
    parser.add_argument("--exclude_preposition_tag", action="store_true")
    args = parser.parse_args()
    return args


def exclude_grammar_terms(words, count=0):
    """exclude under count"""
    new_words = {}
    for key in words:
        if words[key] > count:
            new_words[key] = words[key]
    return new_words


def exclude_preposition_tag_func(words):
    """exclude preposition tag from words dict"""
    return {key: value for key, value in words.items() if key != "<preposition>"}


def use_only_top_freq(words, num=20):
    """use only top num freq"""
    return {key: value for key, value in list(words.items())[-1 * num :]}


def main():
    """main func"""

    args = init_args()

    paths = ["data/train_dev/TRAIN.prep_feedback_comment.public.tsv"]

    words = {}
    n_data = 0
    for path in paths:
        with open(path, encoding="utf-8") as file:
            for line in file:
                n_data += 1
                _, _, target = line.strip().split("\t")
                target = utils.clean_up_data(target)
                grammar_terms = utils.extract_grammar_terms(target)
                for term in grammar_terms:
                    words[term] = words.get(term, 0) + 1

    words = dict(sorted(words.items(), key=lambda x: x[1]))

    with open(args.out, "w", encoding="utf-8") as file:
        file.write("\n".join([k + "\t" + str(v) for k, v in words.items()]))

    words = exclude_grammar_terms(words, count=args.exclude_grammar_terms_by_count)

    if args.exclude_preposition_tag:
        words = exclude_preposition_tag_func(words)

    if args.only_top_n_vocab:
        words = use_only_top_freq(words, args.n_terms)

    print(json.dumps(words, indent=4))
    # print(len(words))

    # print(l, r)

    # print(len(lr_diff))

    # for line in lr_diff:
    # print(line)

    # bef weight
    # n_samples = sum(words.values())
    # weight = []
    # for key in words:
    #     wei = n_samples / (len(words) * words[key])
    # weight.append(wei)

    weights = []
    for key in words:
        wei = math.log(n_data / words[key])
        weights.append(wei)

    out2 = [
        f"exc_{str(args.exclude_grammar_terms_by_count)}",
        f"only_top_n_vocab={str(args.only_top_n_vocab)}",
        f"n_terms={str(args.n_terms)}",
        f"exc_p={str(args.exclude_preposition_tag)}",
    ]
    print(words)
    print(weights)
    print(len(weights))
    with open("data/" + ".".join(out2), "wb") as file:
        pickle.dump(weights, file)


main()
