# Calculates variance in the outputs in the lexical space using WER as distance between two outputs.

import argparse

import utils
from tqdm import tqdm


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--distance_scores", type=str, required=True, help="Model outputs distance scores")
    argparser.add_argument("--output", type=str, required=True)
    return argparser.parse_args()


def _format(results):
    formatted = [",".join(["topic", "concept", "variance"])]
    for topic in results:
        for concept in results[topic]:
            formatted.append(",".join([topic, concept, str(results[topic][concept])]))
    return formatted


def main(distance_scores):
    results = utils.init_results(distance_scores)
    for topic in distance_scores:
        for concept in tqdm(distance_scores[topic]):
            var = 0
            n = len(distance_scores[topic][concept])  # Length of identities
            for identity1 in distance_scores[topic][concept]:
                for identity2 in distance_scores[topic][concept][identity1]:
                    d = distance_scores[topic][concept][identity1][identity2]
                    var += d**2
            results[topic][concept] = var / (n**2)
    return _format(results)


if __name__ == "__main__":
    args = parse_args()
    distance_scores = utils.parse_similarity_csv(args.distance_scores)
    results = main(distance_scores)
    with open(args.output, "w") as f:
        f.write("\n".join(results))
    print("Done", flush=True)
