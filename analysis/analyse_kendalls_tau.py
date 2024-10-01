import argparse
import json
import pdb
import statistics

import utils
from scipy import stats
from tqdm import tqdm


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--similarity", type=str, help="Path to text similarity file.", required=True
    )
    argparser.add_argument(
        "--cultural_distances",
        type=str,
        help="Path to cultural distances file.",
        required=True
    )
    argparser.add_argument("--output", type=str, default="output.tsv")
    argparser.add_argument(
        "--cultural_distance_type",
        type=str,
        help="Type of cultural distance to use.",
        choices=[
            "hofstede_vector_distance",
            "wvs_250_dims_vector_distance",
        ],
        required=True
    )
    argparser.add_argument(
        "--text_similarity_metric", type=str, default="bleu", required=True
    )

    return argparser.parse_args()


def reverse_ranklist(text_similarity_metric, cultural_distance_type):
    if text_similarity_metric == "bleu":
        if "distance" in cultural_distance_type:
            return (
                True,
                False,
            )  # Bleu == similarity; so ranklist orders should be reversed for comparison
        else:
            return (False, False)
    else:
        if "distance" in cultural_distance_type:
            return (
                False,
                False,
            )  # WER == distance; so ranklist orders should be the same for comparison
        else:
            return (True, False)


def get_rank_list(data, identity, reverse):
    raw_data = {k: data[k] for k in data if k != identity}
    unsorted_data = {}
    for k in raw_data:
        if raw_data[k] != None:
            # Removes any None values from the list, None values might appear due to Hofstede data not being available for some countries
            unsorted_data[k] = raw_data[k]
    sorted_data = sorted(unsorted_data.items(), key=lambda x: x[1], reverse=reverse)
    rank_list = [x[0] for x in sorted_data]
    return rank_list


def get_cultural_distance_ranklist(distances, reverse):
    distance_ranklists = {}
    for anchor in distances:
        distance_ranklists[anchor] = get_rank_list(
            distances[anchor], anchor, reverse=reverse
        )
        # print(distances[anchor])
        # print(distance_ranklists[anchor])
    return distance_ranklists


def remove_uncommon(rank_list1, rank_list2):
    rank_list1 = [x for x in rank_list1 if x in rank_list2]
    rank_list2 = [x for x in rank_list2 if x in rank_list1]
    return rank_list1, rank_list2


def _format(results):
    formatted_results = []
    formatted_results.append("topic,concept,anchor,correlation,two-sided-pvalue")
    for topic in results:
        for concept in results[topic]:
            row = [
                topic,
                concept,
                "all_anchors_average",
                str(results[topic][concept]["all_anchors_average"]),
            ]
            formatted_results.append(",".join(row))
            if concept == "all_concepts":
                for anchor in results[topic][concept]:
                    if anchor == "all_anchors_average":
                        continue
                    row = [topic, concept, anchor, str(results[topic][concept][anchor])]
                    formatted_results.append(",".join(row))
    return formatted_results


def main(similarity, distances, reversal):
    results = utils.init_results(similarity)
    distance_rank_lists = get_cultural_distance_ranklist(distances, reversal[0])
    print("created cultural distance ranklists")
    for topic in tqdm(similarity):
        if topic not in results:
            results[topic] = {}
        for concept in tqdm(similarity[topic]):
            if concept not in results[topic]:
                results[topic][concept] = {}
            results[topic][concept]["all_anchors_average"] = []
            for anchor in similarity[topic][concept]:
                if anchor not in distance_rank_lists:
                    continue
                if anchor not in results[topic][concept]:
                    results[topic][concept][anchor] = None
                # print(anchor)
                cultural_distance_ranklist = distance_rank_lists[anchor]
                text_similarity_ranklist = get_rank_list(
                    similarity[topic][concept][anchor], anchor, reverse=reversal[1]
                )

                cultural_distance_ranklist, text_similarity_ranklist = remove_uncommon(
                    cultural_distance_ranklist, text_similarity_ranklist
                )
                assert len(cultural_distance_ranklist) == len(text_similarity_ranklist)

                cultural_distance_ranklist = cultural_distance_ranklist[1:]
                text_similarity_ranklist = text_similarity_ranklist[1:]
                correlation = stats.kendalltau(
                    cultural_distance_ranklist, text_similarity_ranklist, variant="c"
                )
                results[topic][concept][anchor] = correlation.statistic

                results[topic][concept]["all_anchors_average"].append(
                    correlation.statistic
                )
            results[topic][concept]["all_anchors_average"] = statistics.mean(
                results[topic][concept]["all_anchors_average"]
            )
    # print(results)
    return _format(results)


if __name__ == "__main__":
    args = parse_args()
    similarity = utils.parse_similarity_csv(args.similarity)
    print("got text similarity/distances")
    distances = utils.get_distances(args.cultural_distances, args.cultural_distance_type)
    print("got cultural distances")
    reversal = reverse_ranklist(
        args.text_similarity_metric, args.cultural_distance_type
    )
    print("reversal", reversal)
    results = main(similarity, distances, reversal)
    with open(args.output, "w") as f:
        f.write("\n".join(results))
    print("done", flush=True)
