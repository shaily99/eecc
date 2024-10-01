# Extrinsic Evaluation of Cultural Competence in Large Language Models

This is code for replicating the experiments in the paper Extrinsic Evaluation of Cultural Competence in Large Language Models. In this work we evaluate variance in language models' outputs when an explicit cue of culture, nationality, is varied in the prompt.

## Citation

If you use our code or data please cite our paper:

```
@inproceedings{bhatt-diaz-2024-extrinsic,
    title = "Extrinsic Evaluation of Cultural Competence in Large Language Models",
    author = "Bhatt, Shaily  and
      Diaz, Fernando",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida",
    publisher = "Association for Computational Linguistics"
}
```

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

### Model Responses

We make the model outputs public for further analysis on [this huggingface data repository]([test](https://huggingface.co/datasets/shaily99/eecc))

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

### Correlation with Cultural Values

#### Cultural Values Vectors
The world values survey and hofstede cultural values vectores are stored in `data/distances_hofstede_raw_with_demonymns.csv` and `data/wvs_250_dims_with_demonyms.csv` respectively.

#### Computing Similarity in Outputs

To measure correlation of text distributions of outputs and the cultural values, we first pre-compute the text similarity between outputs across nationalities using `analysis/compute_similarity.py`

```
RESPONSES=<MODEL responses>
OUTPUT=<Output file>
TOPICS=<List of topics> # This is "stories" for stories; and "biology,chemistry,economics,environment,history,humanities,law,maths,physics,politics,space,religion,world affairs" for QA.

python analysis/compute_similarity.py --responses ${RESPONSES} --output ${OUTPUT} --topic_keys ${TOPICS} --metric "bleu"
```


#### Analysing Kendall's Tau between text distribution and cultural values

To calculate the correlation for Section 5.3, we use `anaylsis/analyse_kendalls_tau`

```
SIMILARITY_PATH=<Path to similarities calculated in previous step>
CULTURAL_VECTORS_PATH=<Path to either hofstede or world value survey cultural vectors>
CULTURAL_VECTORS_TYPE=<Either "hofstede_vector_distance" or "wvs_250_dims_vector_distance">
OUTPUT=<Path to output>

python analysis/analyse_kendalls_tau.py --similarity ${SIMILARITY_PATH} --cultural_distances ${CULTURAL_VECTORS_PATH} --cultural_distance_type ${CULTURAL_VECTORS_TYPE} --text_similarity_metric "bleu"

```

### Correlated Words

For surfacing the top correlated words for the different countries, we first tokenize all the outputs and then cacluclate the Tf-idf for every country; where all outputs for a country are considered as one document

#### Output tokenization

We use `analysis/store_tokens.py` for tokenizing all outputs. This uses the NLTK word_tokenize() for tokenization.

```
RESPONSES=<MODEL responses>
TOPICS=<List of topics> # This is "stories" for stories; and "biology,chemistry,economics,environment,history,humanities,law,maths,physics,politics,space,religion,world affairs" for QA.

python analysis/store_tokens.py --responses ${RESPONSES} --topic_keys ${TOPICS}
```

#### Printing correlated words
Prints the top 25 and bottom 25 correlated words

```
TOKENIZED_RESPONSES=<Tokenized MODEL responses>
OUTPUT=<Output path>

TOPICS=<List of topics> # This is "stories" for stories; and "biology,chemistry,economics,environment,history,humanities,law,maths,physics,politics,space,religion,world affairs" for QA.

python analysis/store_tokens.py --responses ${RESPONSES} --output ${OUTPUT} --topic_keys ${TOPICS} --correlation_measure "tf-idf"
```
