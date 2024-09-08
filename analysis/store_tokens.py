import argparse

import utils
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=str, help="Model outputs file")
    parser.add_argument("--ig", type=str, default="country")
    parser.add_argument("--topic_keys", type=str, default="")
    return parser.parse_args()


def main(data):
    results = []
    for topic in tqdm(data):
        print(topic, flush=True)
        for concept in data[topic]:
            if concept == "all_concepts":
                continue
            # print(concept, flush=True)
            for identity in data[topic][concept]:
                for template in data[topic][concept][identity]:
                    responses = data[topic][concept][identity][template]
                    tokenized_responses = []
                    for response in responses:
                        tokenized_response = word_tokenize(response)
                        tokenized_responses.append("|_|".join(tokenized_response))
                        # print(row, flush=True)
                    row = [topic, concept, identity, template]
                    row.extend(tokenized_responses)
                    results.append("\t".join(row))
    return results


if __name__ == "__main__":
    args = parse_args()
    print(args, flush=True)
    responses = utils.get_responses(args.responses, args.ig, args.topic_keys)
    output_file = args.responses.replace(".tsv", "_tokenized.tsv")
    results = main(responses)
    with open(output_file, "w") as f:
        f.write("\n".join(results))
