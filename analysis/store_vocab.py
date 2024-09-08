import argparse
import json

import utils
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Store vocab with tf-idf")
    parser.add_argument(
        "--tokenized_responses",
        type=str,
        required=True,
        help="Input file containing tokenized data",
    )
    parser.add_argument("--tfidf_threshold", type=float, default=0.0)
    parser.add_argument("--topic_keys", type=str, default="")

    return parser.parse_args()


def main(tokenized_data, tfidf_threshold):
    country_wise_responses_tokens = {}
    print("Creating overall vocab", flush=True)
    for topic in tqdm(tokenized_data):
        country_wise_responses_tokens[topic] = {}
        for concept in tokenized_data[topic]:
            for identity in tokenized_data[topic][concept]:
                if identity not in country_wise_responses_tokens[topic]:
                    country_wise_responses_tokens[topic][identity] = []
                country_wise_responses_tokens[topic][identity].extend(
                    tokenized_data[topic][concept][identity]
                )
    print("Creating country wise vocab", flush=True)
    country_wise_vocab = {}
    for topic in tqdm(country_wise_responses_tokens):
        country_wise_vocab[topic] = {}
        for country in tqdm(country_wise_responses_tokens[topic]):
            country_wise_vocab[topic][country] = utils.create_vocab(
                country_wise_responses_tokens[topic][country]
            )
            # print(country_wise_vocab[topic][country])
    print("Calculating distributions", flush=True)
    tfidf = {}
    for topic in tqdm(country_wise_vocab):
        tfidf[topic] = {}
        print("Calculating TF-IDF", flush=True)
        for country in country_wise_vocab[topic]:
            sorted_results = utils.tf_idf(
                country_wise_vocab[topic], country, tfidf_threshold
            )
            tfidf[topic][country] = sorted_results
            # print(sorted_results[:5])
    return tfidf, country_wise_vocab


if __name__ == "__main__":
    args = parse_args()
    tokenized_responses = utils.get_response_tokens(
        args.tokenized_responses, args.topic_keys
    )
    tfidf, country_wise_vocab = main(tokenized_responses, args.tfidf_threshold)
    tfidf_path = args.tokenized_responses.replace("tokenized.tsv", "tfidf.json")
    with open(tfidf_path, "w") as f:
        json.dump(tfidf, f, indent=4)
    vocab_path = args.tokenized_responses.replace("tokenized.tsv", "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(country_wise_vocab, f, indent=4)
