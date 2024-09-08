def get_prompts(path):
    with open(path) as f:
        prompts = f.readlines()
    prompts = [prompt.strip() for prompt in prompts]
    header = prompts[0].split("\t")
    prompts = prompts[1:]  # skipping header for now
    return header, prompts


def get_concepts(path, topics):
    with open(path) as f:
        raw_concepts = f.readlines()
    concepts = []
    for line in raw_concepts:
        [topic, concept] = line.split("\t")
        if topic not in topics:
            continue
        concepts.append((topic.strip(), concept.strip()))
    return concepts
