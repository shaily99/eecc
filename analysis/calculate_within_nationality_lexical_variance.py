import utils
import argparse
from tqdm import tqdm
import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--responses", type=str, required=True, help="Path to model responses"
    )
    parser.add_argument("--output", type=str, required=True, help="Path to output file")
    parser.add_argument(
        "--topic_keys", type=str, required=True, help="Topics to analyse"
    )
    parser.add_argument(
        "--granularity",
        type=str,
        required=True,
        help="Granularity of analysis",
        default="country",
        choices=["country"],
    )
    return parser.parse_args()


def _format(results):
    formatted = [",".join(["topic", "concept", "anchor", "variance"])]
    for topic in results:
        for concept in results[topic]:
            if concept == "average":
                formatted.append(
                    ",".join(
                        [topic, "average", "average", str(results[topic][concept])]
                    )
                )
                continue
            formatted.append(
                ",".join(
                    [
                        topic,
                        concept,
                        "all_anchors_average",
                        str(results[topic][concept]),
                    ]
                )
            )
            # for anchor in results[topic][concept]:
            #     formatted.append(
            #         ",".join(
            #             [topic, concept, anchor, str(results[topic][concept][anchor])]
            #         )
            #     )
    return formatted


def main(responses):
    results = utils.init_results(responses)
    for topic in tqdm(responses):
        all_concept_variances = []
        for concept in tqdm(responses[topic]):
            all_anchor_variances = []
            if concept == "all_concepts":
                continue
            for anchor in responses[topic][concept]:
                template = list(responses[topic][concept][anchor].keys())[0]
                anchor_responses = responses[topic][concept][anchor][template]
                variance = 0
                n = len(anchor_responses)
                for r1 in anchor_responses:
                    for r2 in anchor_responses:
                        d = utils.wer(r1, r2, tokenize=True)
                        variance += d**2
                variance = variance / (n**2)
                # results[topic][concept][anchor] = variance
                all_anchor_variances.append(variance)
                all_concept_variances.append(variance)
            results[topic][concept] = sum(all_anchor_variances) / len(
                all_anchor_variances
            )
        results[topic]["average"] = sum(all_concept_variances) / len(
            all_concept_variances
        )
    return _format(results)


if __name__ == "__main__":
    args = parse_args()
    responses = utils.get_responses(args.responses, args.granularity, args.topic_keys)
    print("Got responses", flush=True)
    results = main(responses)
    with open(args.output, "w") as f:
        f.write("\n".join(results))
    print("Done!")
