import argparse

from scipy.stats import binomtest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.path, "r") as f:
        data = f.readlines()
    header = data[0].strip().split(",")
    data = data[1:]
    # header.append("% TRUE")
    header.append("one_sided_p-value")
    header.append("one_sided_significant")
    header.append("two_sided_p-value")
    header.append("two_sided_significant")
    results = [",".join(header)]
    for line in data:
        line = line.strip().split(",")
        n = int(line[7])
        k = int(line[4])
        if n == 0:
            results.append(",".join(line))
            continue
        if k / n > 0.5:
            one_sided_p = binomtest(k, n, 0.5, alternative="greater").pvalue
        else:
            one_sided_p = binomtest(k, n, 0.5, alternative="less").pvalue
        one_sided_significant = one_sided_p < 0.05
        two_sided_p = binomtest(k, n, 0.5).pvalue
        two_sided_significant = two_sided_p < 0.05
        # line.append(str(round(k / n, 4)))
        line.append(str(round(one_sided_p, 4)))
        line.append(str(one_sided_significant))
        line.append(str(round(two_sided_p, 4)))
        line.append(str(two_sided_significant))
        results.append(",".join(line))
    with open(args.output_path, "w") as f:
        f.write("\n".join(results))


if __name__ == "__main__":
    main()
