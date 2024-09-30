# Extrinsic Evaluation of Cultural Competence in Large Language Models

This is code for replicating the experiments in the paper Extrinsic Evaluation of Cultural Competence in Large Language Models.

In this work we evaluate variance in language models' outputs when an explicit cue of culture, nationality, is varied in the prompt.

## Corpus Collection

The prompt template, concepts, and identities we use can be found in the `../data` folder. 

### Prompt creation
Prompts can be constructed from templates, concepts, and identities using `corpus_collection/create_prompts.py`


```
BASE_DIR="data/"
TASKS=("qa" "stories")
IDENTITIES="data/identities.tsv"

for TASK in "${TASKS[@]}"; do
    CONCEPTS="${BASE_DIR}/${TASK}/concepts.tsv
    TEMPLATES="${BASE_DIR}/${TASK}/templates.tsv
    OUTPUT="${BASE_DIR}/${TASK}/prompts.tsv
    
    python corpus_collection/create_prompts.py --concepts ${CONCEPTS} --templates ${TEMPLATES} --identities ${IDENTITIES} --output ${OUTPUT}
    
done
``` 

### Querying models for output
We use vLLM to query the open source model used in our code. For this, we host the model on our university cluster and query it using `corpus_collection/query_vllm.py`

```
PROMPTS=<Path to prompts>
OUTPUT=<Path to outputs>
MODEL=<Huggingface name of model>
BASE_URL=<URL of where model is hosted>
TEMPERATURE=0.3 # we use 0.3 and 0.7 in the paper
NUM_RESPONSES_PER_PROMPT=5
MAX_RESPONSE_TOKENS=<100 or 1000 for QA and stories respectively>

python corpus_collection/query_vllm.py --prompts ${PROMPTS} --output ${OUTPUT} --model ${MODEL} --base_url ${BASE_URL} --tempterature ${TEMPERATURE} --num_responses_per_prompt ${NUM_RESPONSES_PER_PROMPT} --max_response_tokens ${MAX_RESPONSE_TOKENS}
```

## Analysis of Outputs

### Lexical Variance

#### Lexical Variance Across Nationalities

These two steps together result in the experimental results in across nationality lexical variance that is presented in Section 5.1


##### Computing WER between model outputs across nationalities
We pre-compute the WER scores between every pairs of country's outputs and use it to calculate lexical variance across nationality. For calculating WER we use `analysis/compute_similarity.py`. Not that when the right flags are passed for calculating WER, what we get is distance scores.

To compute WER run:

```
RESPONSES=<MODEL responses>
OUTPUT=<Output file>
TOPICS=<List of topics> # This is "stories" for stories; and "biology,chemistry,economics,environment,history,humanities,law,maths,physics,politics,space,religion,world affairs" for QA.

python analysis/compute_similarity.py --responses ${RESPONSES} --output ${OUTPUT} --topic_keys ${TOPICS} --metric "wer"
```

##### Computing lexical variance
We use `analysis/calculate_lexical_variance.py` to calculate the lexical variance outputs

```
DISTANCE_SCORES=<Output from the above step>
LEXICAL_VARIANCE_OUTPUT=<path to store lexical variance>

python analysis/calculate_lexical_variance.py --distance_scores ${DISTANCE_SCORES} --output ${LEXICAL_VARIANCE_OUTPUT}
```

##### Significance testing

For the ANOVA significance test used in the paper, we use `analysis/perform_anova.py`


#### Within Nationality Variance

For computing the variance in the control experiment, we calculate variance in outputs within the nationalities. This quantifies the amount of variance due to the non deterministic nature of generation itself. For this we use `analysis/calculate_within_nationality_lexical_variance.py`

```
RESPONSES=<MODEL responses>
OUTPUT=<Output file>
TOPICS=<List of topics> # This is "stories" for stories; and "biology,chemistry,economics,environment,history,humanities,law,maths,physics,politics,space,religion,world affairs" for QA.

python analysis/calculate_within_nationality_lexical_variance.py --responses ${RESPONSES} --output ${OUTPUT} --topic_keys ${TOPICS}

```

