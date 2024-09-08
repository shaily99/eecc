from scipy.stats import f_oneway
import pandas as pd
import pdb

base_dir = [
    "/home/shailyjb/pref_cult/creative_prompts",
    "/home/shailyjb/pref_cult/hand_curated_large",
]

models = [
    "llama2_7B_chat_vllm",
    "llama2_13B_chat_vllm",
    "llama3_8B_instruct_vllm",
    "llama3_8B_instruct",
    "gemma2B_it",
    "gemma7B_it",
    "gpt_3-5_20240328",
    "gpt_3-5_20240223-0310",
]


def get_within_nationality_variance(path):
    variance_values = []
    with open(path, "r") as f:
        data = f.readlines()
    for line in data:
        if "," not in line:
            pdb.set_trace()
        line = line.replace("gpt 3.5,", "")
        [topic, concept, anchor, value] = line.strip().split(",")
        if value == "{}":
            continue
        if concept == "all_concepts":
            continue
        if anchor == "all_anchors_average":
            try:
                variance_values.append(float(value))
            except:
                pdb.set_trace()
    return variance_values


def get_across_nationality_variance(path):
    variance_values = []
    with open(path, "r") as f:
        data = f.readlines()
    for line in data:
        [topic, concept, value] = line.strip().split(",")
        if value == "variance":
            continue
        if concept == "all_concepts":
            continue
        try:
            variance_values.append(float(value))
        except:
            pdb.set_trace()
    return variance_values


print(
    "\\textbf{task} & \\textbf{model} & \\textbf{statistic} & \\textbf{p-value} & \\textbf{Reject H0} \\\\ \hline"
)
for base in base_dir:
    for model in models:
        if "creative_prompts" in base:
            if model == "gpt_3-5_20240223-0310":
                continue
            if model == "llama3_8B_instruct_vllm":
                continue
            result_dir = f"{base}/results/{model}/wer/country"
            within_nationality_variance_path = (
                f"{result_dir}/within_nationality_variance.csv"
            )
            across_nationality_variance_path = (
                f"{result_dir}/lexical_variance_stories_combined.csv"
            )
        elif "hand_curated_large" in base:
            if model == "gpt_3-5_20240328":
                continue
            if model == "llama3_8B_instruct":
                continue
            result_dir = f"{base}/results/100_tokens/{model}/wer/country"
            across_nationality_variance_path = f"{result_dir}/lexical_variance.csv"
            within_nationality_variance_path = (
                f"{result_dir}/within_nationality_variance.csv"
            )
        else:
            pdb.set_trace()

        within_nationality_variances = get_within_nationality_variance(
            within_nationality_variance_path
        )
        across_nationality_variances = get_across_nationality_variance(
            across_nationality_variance_path
        )

        assert len(within_nationality_variances) == len(across_nationality_variances)

        anova_result = f_oneway(
            within_nationality_variances, across_nationality_variances
        )
        reject = (
            "\\textcolor{teal}{Yes}"
            if anova_result.pvalue < 0.05
            else "\\textcolor{red}{No}"
        )
        title = "stories" if "creative_prompts" in base else "QA"
        print(
            f"{title} & {model} & {round(anova_result.statistic, 4)} & {anova_result.pvalue} & {reject} \\\\"
        )
