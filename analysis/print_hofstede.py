import utils

hofstede_path = "/home/shailyjb/pref_cult/hofstedes_distance/distances_hofstede_raw_with_demonymns.csv"
distances = utils.get_hofstede_distances(hofstede_path, type="hofstede_vector_distance")

countries = sorted(list(distances.keys()))

header_row = "," + ",".join(countries)
results = [header_row]
for country1 in countries:
    row = [country1]
    for country2 in countries:
        if distances[country1][country2] is None:
            row.append("None")
        else:
            row.append(str(distances[country1][country2]))
    results.append(",".join(row))

output_path = (
    "/home/shailyjb/pref_cult/hofstedes_distance/pairwise_hofstede_vector_distances.csv"
)
with open(output_path, "w") as file:
    file.write("\n".join(results))
