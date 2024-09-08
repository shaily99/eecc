# Analyse the word distributions for the outputs using KL Divergence

import argparse
import math

import nltk
import utils
from tqdm import tqdm

nltk.download("stopwords")
# print(stopwords.words("english"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenized_model_outputs", type=str, help="Tokenized model outputs file"
    )
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument(
        "--level",
        type=str,
        help="Level of analysis. If empty, all topics are combined and a single word distribution is analysed. Otherwise word distribution is analysed at topic level.",
        choices=["topic", ""],
        default="",
    )
    parser.add_argument(
        "--topic_keys",
        type=str,
        help="Comma separated list of topics to analyse",
        default="",
    )
    parser.add_argument(
        "--type",
        type=str,
        help="Type of analysis",
        default="tfidf",
        choices=["kld", "tfidf"],
    )
    return parser.parse_args()


def word_wise_kl_divergence(p, q):
    results = {}
    for word in p:
        if word in q:
            results[word] = p[word] * math.log((p[word] / q[word]))
    # Get only the top 25 and bottom 25 words
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:25], sorted_results[-25:]


def _format(results):
    formatted = []
    for country in results:
        row = []
        row.append(country)
        row.append("top")
        for word, score in results[country]["top"]:
            row.append(word)
            row.append(str(score))
        formatted.append("\t".join(row))
        row = []
        row.append(country)
        row.append("bottom")
        for word, score in results[country]["bottom"]:
            row.append(word)
            row.append(str(score))
        formatted.append("\t".join(row))
    return formatted


def _format_topic_wise(results):
    print("Formatting results", flush=True)
    formatted = []
    for topic in results:
        for country in results[topic]:
            row = []
            row.append(topic)
            row.append(country)
            row.append("top")
            for word, score in results[topic][country]["top"]:
                row.append(word)
                row.append(str(score))
            formatted.append("\t".join(row))
            row = []
            row.append(topic)
            row.append(country)
            row.append("bottom")
            for word, score in results[topic][country]["bottom"]:
                row.append(word)
                row.append(str(score))
            formatted.append("\t".join(row))
    return formatted


def main_overall(tokenized_data, type):
    all_responses_tokens = []
    country_wise_responses_tokens = {}
    print("creating vocab", flush=True)
    for topic in tqdm(tokenized_data):
        for concept in tokenized_data[topic]:
            for identity in tokenized_data[topic][concept]:
                if identity not in country_wise_responses_tokens:
                    country_wise_responses_tokens[identity] = []
                country_wise_responses_tokens[identity].extend(
                    tokenized_data[topic][concept][identity]
                )
                all_responses_tokens.extend(tokenized_data[topic][concept][identity])
    print("Creating country wise vocab", flush=True)
    country_wise_vocab = {}
    for country in tqdm(country_wise_responses_tokens):
        country_wise_vocab[country] = utils.create_vocab(
            country_wise_responses_tokens[country]
        )
    print("Calculating distributions", flush=True)
    results = {}
    if type == "kld":
        overall_probs = utils.calculate_vocab_probs(utils.create_vocab(all_responses_tokens))
        print("Calculating KL Divergence", flush=True)
        for country in tqdm(country_wise_vocab):
            top, bottom = word_wise_kl_divergence(
                utils.calculate_vocab_probs(country_wise_vocab[country]), overall_probs
            )
            results[country] = {"top": top, "bottom": bottom}
    elif type == "tfidf":
        print("Calculating TF-IDF", flush=True)
        for country in tqdm(country_wise_vocab):
            sorted_results = utils.tf_idf(country_wise_vocab, country)
            top = sorted_results[:25]
            bottom = sorted_results[-25:]
            results[country] = {"top": top, "bottom": bottom}
    return _format(results)


def main_topic(tokenized_data, type):
    all_responses_tokens = {}
    country_wise_responses_tokens = {}
    print("Creating overall vocab", flush=True)
    for topic in tqdm(tokenized_data):
        all_responses_tokens[topic] = []
        country_wise_responses_tokens[topic] = {}
        for concept in tokenized_data[topic]:
            for identity in tokenized_data[topic][concept]:
                if identity not in country_wise_responses_tokens[topic]:
                    country_wise_responses_tokens[topic][identity] = []
                country_wise_responses_tokens[topic][identity].extend(
                    tokenized_data[topic][concept][identity]
                )
                all_responses_tokens[topic].extend(
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
    results = {}
    print("Calculating distributions", flush=True)
    for topic in tqdm(country_wise_vocab):
        results[topic] = {}
        if type == "kld":
            overall_probs = utils.calculate_vocab_probs(
                utils.create_vocab(all_responses_tokens[topic])
            )
            print("Calculating KL Divergence", flush=True)
            for country in country_wise_vocab[topic]:
                top, bottom = word_wise_kl_divergence(
                    utils.calculate_vocab_probs(country_wise_vocab[topic][country]),
                    overall_probs,
                )
                results[topic][country] = {"top": top, "bottom": bottom}
        elif type == "tfidf":
            print("Calculating TF-IDF", flush=True)
            for country in country_wise_vocab[topic]:
                sorted_results = utils.tf_idf(country_wise_vocab[topic], country)
                top = sorted_results[:25]
                bottom = sorted_results[-25:]
                results[topic][country] = {"top": top, "bottom": bottom}
    return _format_topic_wise(results)


if __name__ == "__main__":
    args = parse_args()
    tokenized_data = utils.get_response_tokens(
        args.tokenized_model_outputs, args.topic_keys
    )
    if args.level == "topic":
        print("Topic level analysis", flush=True)
        results = main_topic(tokenized_data, args.type)
    else:
        print("Overall analysis", flush=True)
        results = main_overall(tokenized_data, args.type)
    with open(args.output, "w") as f:
        f.write("\n".join(results))
