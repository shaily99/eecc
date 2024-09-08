# Computes similarity between responses for each concept and identity pair.

import argparse
import pdb

import utils
from tqdm import tqdm


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--responses", type=str, default="gpt_3-5_responses.tsv")
    argparser.add_argument("--ig", type=str, default="country")
    argparser.add_argument("--metric", type=str, default="bleu")
    argparser.add_argument("--output", type=str, default="output.tsv")
    argparser.add_argument("--topic_keys", type=str, required=True)

    return argparser.parse_args()


def _format(results):
    print("Formatting")
    formatted_results = []
    topics = list(results.keys())
    print(len(topics))
    topics.sort()
    for topic in topics:
        formatted_results.append(f"\n\n***** TOPIC: {topic} *****\n")
        concepts = list(results[topic].keys())
        concepts.sort()
        for concept in concepts:
            formatted_results.append(f"\n***** Concept: {concept} *****\n")
            identities = list(results[topic][concept].keys())
            identities.sort()
            formatted_results.append("," + ",".join(identities))
            for identity1 in identities:
                if identity1 not in results[topic][concept]:
                    results[topic][concept][identity1] = {}
                    print("Something went wrong.")
                row = [identity1]
                for identity2 in identities:
                    if identity2 not in results[topic][concept][identity1]:
                        results[topic][concept][identity1][
                            identity2
                        ] = "No similarity score for this pair."
                        print(
                            "Something went wrong. 2.",
                            identity1,
                            identity2,
                            topic,
                            concept,
                        )
                    row.append(
                        str(
                            results[topic][concept][identity1][identity2],
                        )
                    )
                formatted_results.append(",".join(row))
            formatted_results.append("\n")
        formatted_results.append("\n")
        print("Done formatting")
    return formatted_results



def main(metric, data):
    print("main", flush=True)
    results = utils.init_results(data)
    print("init_results done", flush=True)
    print(results, flush=True)
    print("starting similarity computation", flush=True)
    for topic in tqdm(data):
        all_concepts_similarity = {}
        ctr = {}
        for concept in tqdm(data[topic]):
            # print(concept)
            if concept == "all_concepts":
                continue
            identities = list(data[topic][concept].keys())
            # print(identities, flush=True)
            # for identity1 in tqdm(identities):
            for identity1 in identities:
                for identity2 in identities:
                    # print(identity1, identity2, flush=True)
                    if identity1 not in results[topic][concept]:
                        results[topic][concept][identity1] = {}
                    if identity1 not in all_concepts_similarity:
                        all_concepts_similarity[identity1] = {}
                    if identity1 not in ctr:
                        ctr[identity1] = {}
                    if identity2 not in results[topic][concept]:
                        results[topic][concept][identity2] = {}
                    if identity2 not in all_concepts_similarity:
                        all_concepts_similarity[identity2] = {}
                    if identity2 not in ctr:
                        ctr[identity2] = {}
                    if identity2 not in results[topic][concept][identity1]:
                        results[topic][concept][identity1][
                            identity2
                        ] = "No common templates."
                    if identity1 not in results[topic][concept][identity2]:
                        results[topic][concept][identity2][
                            identity1
                        ] = "No common templates."
                    if identity2 not in ctr[identity1]:
                        ctr[identity1][identity2] = 0
                    if identity1 not in ctr[identity2]:
                        ctr[identity2][identity1] = 0
                    if identity2 not in all_concepts_similarity[identity1]:
                        all_concepts_similarity[identity1][identity2] = 0
                    if identity1 not in all_concepts_similarity[identity2]:
                        all_concepts_similarity[identity2][identity1] = 0

                    if (
                        identity2 in results[topic][concept][identity1]
                        and results[topic][concept][identity1][identity2]
                        != "No common templates."
                    ):
                        try:
                            assert (
                                results[topic][concept][identity1][identity2]
                                == results[topic][concept][identity2][identity1]
                            )
                            assert (
                                all_concepts_similarity[identity1][identity2]
                                == all_concepts_similarity[identity2][identity1]
                            )
                            assert (
                                ctr[identity1][identity2] == ctr[identity2][identity1]
                            )
                        except AssertionError:
                            pdb.set_trace()
                        continue

                    similarity_score = utils.compute_similarity(
                        data[topic][concept][identity1],
                        data[topic][concept][identity2],
                        metric,
                    )
                    if similarity_score is not None:
                        results[topic][concept][identity1][identity2] = similarity_score
                        results[topic][concept][identity2][identity1] = similarity_score
                        ctr[identity1][identity2] += 1
                        ctr[identity2][identity1] += 1
                        all_concepts_similarity[identity1][
                            identity2
                        ] += similarity_score
                        all_concepts_similarity[identity2][
                            identity1
                        ] += similarity_score
        if "all_concepts" not in results[topic]:
            results[topic]["all_concepts"] = {}
        for identity1 in all_concepts_similarity:
            if identity1 not in results[topic]["all_concepts"]:
                results[topic]["all_concepts"][identity1] = {}
            for identity2 in all_concepts_similarity[identity1]:
                if ctr[identity1][identity2] > 0:
                    results[topic]["all_concepts"][identity1][identity2] = (
                        all_concepts_similarity[identity1][identity2]
                        / ctr[identity1][identity2]
                    )
                else:
                    results[topic]["all_concepts"][identity1][
                        identity2
                    ] = "No common templates."
    # print(results, flush=True)
    return _format(results)


if __name__ == "__main__":
    args = parse_args()
    responses = utils.get_responses(args.responses, args.ig, args.topic_keys)
    print("got_responses", flush=True)
    # print(responses, flush=True)
    results = main(args.metric, responses)
    print(len(results))
    with open(args.output, "w") as f:
        f.write("\n".join(results))
    print("done", flush=True)
