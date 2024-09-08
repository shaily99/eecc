import math
import pdb
from statistics import mean, stdev

import editdistance as ed
import evaluate
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Using editdistance library for WER calculation rather then hf evaluate because hf evaluate does not have support for custom tokenization and simply splits by space.

nltk.download("punkt")

bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")


def get_common_template(responses_id1, responses_id2):
    common_templates = []
    for template in responses_id1:
        if template in responses_id2:
            common_templates.append(template)
    return common_templates


def calculate_bleu(responses_id1, responses_id2):
    common_templates = get_common_template(responses_id1, responses_id2)
    if len(common_templates) == 0:
        print("No common templates. Check Data.", flush=True)
        return -1
    predictions = []
    references = []

    for template in common_templates:
        for p in responses_id1[template]:
            if len(p) == 0:
                continue
            all_references = responses_id2[template]
            all_references = [x for x in all_references if len(x) != 0]
            if len(all_references) == 0:
                continue
            predictions.append(p)
            references.append(all_references)
    if len(predictions) == 0:
        print("No predictions. Check Data.", flush=True)
        return -1
    assert len(predictions) == len(references)

    score1 = bleu.compute(
        predictions=predictions, references=references, tokenizer=word_tokenize
    )
    predictions = []
    references = []
    for template in common_templates:
        for p in responses_id1[template]:
            if len(p) == 0:
                continue
            all_references = responses_id2[template]
            all_references = [x for x in all_references if len(x) != 0]
            if len(all_references) == 0:
                continue
            predictions.append(p)
            references.append(all_references)
    assert len(predictions) == len(references)
    score2 = bleu.compute(
        predictions=predictions, references=references, tokenizer=word_tokenize
    )
    score = mean([score1["bleu"], score2["bleu"]])
    return score


def calculate_bertscore(responses_id1, responses_id2):
    common_templates = get_common_template(responses_id1, responses_id2)
    if len(common_templates) == 0:
        print("No common templates. Check Data.", flush=True)
        return -1
    predictions = []
    references = []

    for template in common_templates:
        for p in responses_id1[template]:
            if len(p) == 0:
                continue
            all_references = responses_id2[template]
            all_references = [x for x in all_references if len(x) != 0]
            if len(all_references) == 0:
                continue
            predictions.append(p)
            references.append(all_references)
    assert len(predictions) == len(references)
    score1 = bertscore.compute(
        predictions=predictions, references=references, lang="en"
    )
    score1 = mean([s for s in score1["f1"]])
    predictions = []
    references = []
    for template in common_templates:
        for p in responses_id1[template]:
            if len(p) == 0:
                continue
            all_references = responses_id2[template]
            all_references = [x for x in all_references if len(x) != 0]
            if len(all_references) == 0:
                continue
            predictions.append(p)
            references.append(all_references)
    assert len(predictions) == len(references)
    score2 = bertscore.compute(
        predictions=predictions, references=references, lang="en"
    )
    score2 = mean([s for s in score2["f1"]])
    score = mean([score1, score2])
    return score


def wer(pred, ref, tokenize=False):
    if tokenize:
        pred = word_tokenize(pred)
        ref = word_tokenize(ref)
    return ed.eval(pred, ref) / max(len(pred), len(ref))


def calculate_wer(responses_id1, responses_id2):
    if responses_id1 == responses_id2:
        return 0
    common_templates = get_common_template(responses_id1, responses_id2)
    if len(common_templates) == 0:
        print("No common templates. Check Data.", flush=True)
        return -1
    for template in common_templates:
        for x in range(len(responses_id1[template])):
            if type(responses_id1[template][x]) == list:
                continue
            responses_id1[template][x] = word_tokenize(responses_id1[template][x])
        for x in range(len(responses_id2[template])):
            if type(responses_id2[template][x]) == list:
                continue
            responses_id2[template][x] = word_tokenize(responses_id2[template][x])
    score = 0
    for template in common_templates:
        score1 = 0
        ctr = 0
        for pred in responses_id1[template]:
            for ref in responses_id2[template]:
                score1 += wer(pred, ref)
                ctr += 1
        score1 = score1 / ctr
        score2 = 0
        ctr = 0
        for pred in responses_id2[template]:
            for ref in responses_id1[template]:
                score2 += wer(pred, ref)
                ctr += 1
        score2 = score2 / ctr
        score += mean([score1, score2])
    return score / len(common_templates)


def compute_similarity(responses_id1, responses_id2, metric):
    if metric == "bleu":
        return calculate_bleu(responses_id1, responses_id2)
    elif metric == "wer":
        return calculate_wer(responses_id1, responses_id2)
    elif metric == "bertscore":
        return calculate_bertscore(responses_id1, responses_id2)
    else:
        raise NotImplementedError("Metric not implemented")


def get_responses(path, ig, topic_keys):
    with open(path, "r") as f:
        responses = f.readlines()
    topic_keys = [topic.strip() for topic in topic_keys.split(",")]
    print(topic_keys, flush=True)
    data = {}
    for idx, response in enumerate(responses):
        if "\t" not in response or (
            "not-brief" not in response and "brief" not in response
        ):
            print(idx, response, flush=True)
            continue
        data_point = response.strip().split("\t")
        data_point = [x.strip() for x in data_point]
        data_point = data_point[2:]
        # if len(data_point) == 0:
        #     pdb.set_trace()
        [template, topic, concept, granularity, identity] = data_point[:5]
        model_responses = data_point[6:]
        if granularity != ig:
            continue
        if topic not in topic_keys:
            continue
        if topic not in data:
            data[topic] = {}
        if concept not in data[topic]:
            data[topic][concept] = {}
        if identity not in data[topic][concept]:
            data[topic][concept][identity] = {}

        template = template.format(
            concept=concept, vowel="[vowel]", identity="[identity]"
        )

        if template not in data[topic][concept][identity]:
            data[topic][concept][identity][template] = []

        for model_response in model_responses:
            data[topic][concept][identity][template].append(
                model_response.replace("~| ", "\n").lower()
            )
    return data


def get_response_tokens(path, topics):
    with open(path, "r") as f:
        responses = f.readlines()
    topics = [topic.strip() for topic in topics.split(",")]
    print(topics, flush=True)
    data = {}
    for response in responses:
        # if len(response.strip().split("\t")) != 6:
        #     print("Wrong length of response. Check Data point: ", flush=True)
        #     continue
        data_point = response.strip().split("\t")
        topic, concept, identity, template = data_point[:4]
        tokenized_responses = data_point[4:]
        if topic not in topics:
            continue
        if topic not in data:
            data[topic] = {}
        if concept not in data[topic]:
            data[topic][concept] = {}
        if identity not in data[topic][concept]:
            data[topic][concept][identity] = []
        for tokenized_response in tokenized_responses:
            data[topic][concept][identity].append(
                tokenized_response.strip().split("|_|")
            )

    return data


def get_physical_distances(path):
    distances = {}
    with open(path, "r") as f:
        dist_data = f.readlines()
    for distance in dist_data:
        [r1, r2, dist] = distance.strip().split(",")
        if r1 not in distances:
            distances[r1] = {r1: 0}
        if r2 not in distances:
            distances[r2] = {r2: 0}
        distances[r1][r2] = float(dist)
        distances[r2][r1] = float(dist)
    return distances


def _vector_distance(v1, v2):
    return sum([(x - y) ** 2 for x, y in zip(v1, v2)]) ** 0.5


def _cosine_similarity(v1, v2):
    assert len(v1) == len(v2)
    assert len(v1) != 0
    assert len(v2) != 0
    dot_product = sum([x * y for x, y in zip(v1, v2)])
    magnitude_v1 = sum([x**2 for x in v1]) ** 0.5
    magnitude_v2 = sum([x**2 for x in v2]) ** 0.5
    if dot_product != 0:
        assert magnitude_v1 != 0
        assert magnitude_v2 != 0
        return dot_product / (magnitude_v1 * magnitude_v2)
    else:
        return 0


def _minus_50_transform(x):
    for i in range(len(x)):
        if x[i] is not None:
            x[i] -= 50
    return x


def _calc_mu_sigma(distances):
    pdi = []
    idv = []
    mas = []
    uai = []
    ltowvs = []
    ivr = []
    for country in distances:
        pdi.append(distances[country][0])
        idv.append(distances[country][1])
        mas.append(distances[country][2])
        uai.append(distances[country][3])
        ltowvs.append(distances[country][4])
        ivr.append(distances[country][5])
    pdi = [x for x in pdi if x is not None]
    idv = [x for x in idv if x is not None]
    mas = [x for x in mas if x is not None]
    uai = [x for x in uai if x is not None]
    ltowvs = [x for x in ltowvs if x is not None]
    ivr = [x for x in ivr if x is not None]
    mu = [mean(pdi), mean(idv), mean(mas), mean(uai), mean(ltowvs), mean(ivr)]
    sigma = [stdev(pdi), stdev(idv), stdev(mas), stdev(uai), stdev(ltowvs), stdev(ivr)]
    return mu, sigma


def _mu_sigma_transform(x, mu, sigma):
    for i in range(len(x)):
        if x[i] is not None:
            x[i] = (x[i] - mu[i]) / sigma[i]
    return x


def transform(raw_distances, transform_type):
    if transform_type == "minus50":
        print("Using -50 transformation", transform_type, flush=True)
        return {
            country: _minus_50_transform(dist)
            for country, dist in raw_distances.items()
        }
    elif transform_type == "musigma":
        mu, sigma = _calc_mu_sigma(raw_distances)
        print("Using mu sigma transformation", transform_type, flush=True)
        return {
            country: _mu_sigma_transform(dist, mu, sigma)
            for country, dist in raw_distances.items()
        }
    else:
        print("No transformation", transform_type, flush=True)
        return raw_distances


def get_wvs_250_dims_distance(path, type):
    wvs_vectors = {}
    with open(path, "r") as f:
        lines = f.readlines()
    lines = lines[1:]  # removing header
    for line in lines:
        line = line.strip().split(",")
        country_code = line[0]
        country = line[1]
        demonym = line[2]
        values = line[3:]
        wvs_vectors[demonym] = [
            float(dist) if dist != "#NULL!" else None for dist in values
        ]

    if "musigma" in type:
        print("Using mu sigma transformation", type, flush=True)
        wvs_vectors = transform(wvs_vectors, "musigma")
    elif "minus50" in type:
        print("Using -50 transformation", type, flush=True)
        wvs_vectors = transform(wvs_vectors, "minus50")
    else:
        print("No transformation", type, flush=True)

    distances = {}
    for country1 in wvs_vectors:
        for country2 in wvs_vectors:
            if country1 not in distances:
                distances[country1] = {country1: 0}
            if country2 not in distances:
                distances[country2] = {country2: 0}
            d1_limited = []
            d2_limited = []
            for dist1, dist2 in zip(wvs_vectors[country1], wvs_vectors[country2]):
                if dist1 is not None and dist2 is not None:
                    d1_limited.append(dist1)
                    d2_limited.append(dist2)
            if len(d1_limited) == 0 or len(d2_limited) == 0:
                distance = None
            else:
                assert len(d1_limited) == len(d2_limited)
                assert len(d1_limited) != 0
                assert len(d2_limited) != 0
                if "cosine" in type:
                    distance = _cosine_similarity(d1_limited, d2_limited)
                else:
                    distance = _vector_distance(d1_limited, d2_limited)
            distances[country1][country2] = distance
            distances[country2][country1] = distance
    return distances


def get_hofstede_distances(path, type):
    hofstede_vectors = {}
    with open(path, "r") as f:
        lines = f.readlines()
    lines = lines[1:]  # removing header
    for line in lines:
        [
            country_code,
            country,
            demonym,
            pdi,
            idv,
            mas,
            uai,
            ltowvs,
            ivr,
        ] = line.strip().split(",")
        hofstede_vectors[demonym] = [
            float(dist) if dist != "#NULL!" else None
            for dist in [pdi, idv, mas, uai, ltowvs, ivr]
        ]

    if "musigma" in type:
        print("Using mu sigma transformation", type, flush=True)
        hofstede_vectors = transform(hofstede_vectors, "musigma")
    elif "minus50" in type:
        print("Using -50 transformation", type, flush=True)
        hofstede_vectors = transform(hofstede_vectors, "minus50")
    else:
        print("No transformation", type, flush=True)

    distances = {}
    for country1 in hofstede_vectors:
        for country2 in hofstede_vectors:
            if country1 not in distances:
                distances[country1] = {country1: 0}
            if country2 not in distances:
                distances[country2] = {country2: 0}
            d1_limited = []
            d2_limited = []
            for dist1, dist2 in zip(
                hofstede_vectors[country1], hofstede_vectors[country2]
            ):
                if dist1 is not None and dist2 is not None:
                    d1_limited.append(dist1)
                    d2_limited.append(dist2)
            if len(d1_limited) == 0 or len(d2_limited) == 0:
                distance = None
            else:
                assert len(d1_limited) == len(d2_limited)
                assert len(d1_limited) != 0
                assert len(d2_limited) != 0
                if "cosine" in type:
                    distance = _cosine_similarity(d1_limited, d2_limited)
                else:
                    distance = _vector_distance(d1_limited, d2_limited)
            distances[country1][country2] = distance
            distances[country2][country1] = distance
    return distances


def get_distances(path, type="physical"):
    if type == "physical":
        return get_physical_distances(path)
    elif "hofstede" in type:
        print("Using Hofstede distances", type, flush=True)
        return get_hofstede_distances(path, type)
    elif "wvs_250_dims" in type:
        print("Using WVS 250 dim distances", type, flush=True)
        return get_wvs_250_dims_distance(path, type)


def init_results(data):
    results = {}
    for topic in data:
        results[topic] = {"all_concepts": {}}
        for concept in data[topic]:
            results[topic][concept] = {}
    return results


def parse_similarity_csv(path):
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    data = {}
    index = 0
    header = None
    while index < len(lines):
        line = lines[index]
        if line.startswith("***** TOPIC:"):
            # New topic starts
            topic = line.split(": ")[1].split(" *****")[0].strip()
            # print("topic: ", topic, flush=True)
            if topic in data:
                print("OVERWRITE. Check ", topic, flush=True)
            data[topic] = {}
            index = index + 1
        elif line.startswith("***** Concept:"):
            # New concept starts
            concept = line.split(": ")[1].split(" *****")[0].strip()
            # print("concept: ", concept, flush=True)
            if concept in data[topic]:
                print("OVERWRITE. Check ", concept, flush=True)
            data[topic][concept] = {}
            index = index + 1
        else:
            # Either data or header row
            row = line.split(",")
            if row[0] == "":
                # Header row
                header = row[1:]
                for header_identity in header:
                    if header_identity in data[topic][concept]:
                        print("OVERWRITE. Check ", header_identity, flush=True)
                    data[topic][concept][header_identity] = {}
                index = index + 1
            else:
                # Data row
                identity = row[0]
                values = row[1:]
                for header_identity, value in zip(header, values):
                    try:
                        data_value = float(value)
                    except ValueError:
                        print(
                            "Value not a number. Data mismatch error in original distance score calculation ",
                            topic,
                            concept,
                            identity,
                            flush=True,
                        )
                        data_value = 0
                    data[topic][concept][header_identity][identity] = data_value

                index = index + 1
    return data


def tf_idf(country_wise_counts, country, tfidf_threshold=0):
    tf_idf_scores = {}
    for word in country_wise_counts[country]:
        tf = country_wise_counts[country][word]
        idf_denominator = 0
        for c in country_wise_counts:
            if word in country_wise_counts[c]:
                idf_denominator += 1
        idf = math.log(len(country_wise_counts) / idf_denominator)
        tf_idf_score = tf * idf
        if tf_idf_score <= tfidf_threshold:
            continue
        tf_idf_scores[word] = tf_idf_score
    sorted_results = sorted(tf_idf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def create_vocab(tokenized_responses):
    vocab = {}
    for tokenized_response in tokenized_responses:
        for token in tokenized_response:
            token = token.strip()
            if len(token) < 2:
                continue
            if token.isnumeric():
                continue
            if (
                token == "'"
                or token == "''"
                or token == "``"
                or token == "`"
                or token == "’"
            ):
                continue
            if token in stopwords.words("english"):
                continue
            try:
                float(token)  # removes any tokens that are numbers
            except ValueError:
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1
    return vocab


def calculate_vocab_probs(vocab):
    total = sum(vocab.values())
    # print(total)
    probs = {}
    for word in vocab:
        probs[word] = vocab[word] / total
    return probs
