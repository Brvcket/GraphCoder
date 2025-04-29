# Towards Better Handling of Large Code Contexts: Improvements to Graph-based Generation

## Overview

Fork with fixed bugs from GraphCoder and integration of various distance metrics with different embedders

## Project Structure

The structure of this project is shown as follows:

```
├─ RepoEval-Updated    # Input dataset for code completion tasks
    ├─ api_level.python.test.jsonl
    ├─ api_level.java.test.jsonl
    ├─ line_level.python.test.jsonl
    └─ line_level.java.test.jsonl
├─ repositories    # The original repositories that RepoEval-Updated built from
    ├─ devchat
    ├─ nemo_aligner
    ├─ awslabs_fortuna
    ├─ task_weaver
    ├─ huggingface_diffusers
    ├─ opendilab_ACE
    ├─ metagpt
    ├─ nerfstudio-project_nerfstudio
    ├─ apple_axlearn
    ├─ QingruZhang_AdaLoRA
    ├─ itlemon_chatgpt4j
    ├─ Aelysium-Group_rusty-connector
    ├─ neoforged_NeoGradle
    ├─ mybatis-flex_mybatis-flex
    ├─ Guiqu1aixi_rocketmq
    ├─ SimonHalvdansson_Harmonic-HN
    ├─ Open-DBT_open-dbt
    ├─ QuasiStellar_custom-pixel-dungeon
    ├─ gentics_cms-oss
    └─ FloatingPoint-MC_MIN
├─ utils
    ├─ __init__.py
    ├─ ccg.py    # Generate a code context graph (CCG) from Python code snippets
    ├─ slicing.py    # Generate CCG skuce and its corresponding contet sequence slice
    ├─ build_prompt.py    # Construct prompt with/without retrieval code snippets for code LLMs
    ├─ metrics.py    # Metric calculation for evaluating retrieval and generation effectiveness
    └─ utils.py    # Other tools for tokenizer and file reading/writing
├─ build_graph_database.py    # Build a key-value database for retrieval
├─ build_query_graph.py    # Generate sliced query CCG
├─ search_code.py    # Coarse-to-fine code retrieval
├─ generate_response.py    # Generate the predicted statement based on retrieval results
├─ requirements.txt    # List of packages required to run GraphCoder
└─ my-languages.so    # Dependence file for code context graph generation (tree-sitter parser for python and java)
```

## Quick Start

Example of usage can be found in bash.ipynb

#### Step 1: Install Requirements

```
pip install -r requirements.txt
```

The generation of code context graph is based on tree-sitter
```
git clone https://github.com/tree-sitter/tree-sitter-python
```
```
git clone https://github.com/tree-sitter/tree-sitter-java
```
#### Step 2: Database Construction

```
python build_graph_database.py
```

### Step 3: Code Retrieval

There are 3 input arguments for code retrieval step:

  - query_cases: 
    - `api_level.python/java`: Run code retrieval for api-level python/java code completion tasks.
    - `line_level.python/java`: Run code retrieval for line-level python/java code completion tasks.

  - mode:
    - `coarse2fine`: This mode corresponds to the code retrieval method in GraphCoder, which performs coarse-grained retrieval and fine-grained re-ranking.
    - `coarse`: This mode corresponds to the variant of GraphCoder, namely GraphCoder-C, which only performs coarse-grained retrieval.
    - `fine`: This mode corresponds to the variant of GraphCoder, namely GraphCoder-F, which performs retrieval only by the fine-grained graph measure (i.e., decay-with-distance subgraph edit distance).

  - gamma: The decay-with-distance factor used in fine-grained step.
    
An example for running code retrieval

```
python search_code.py --query_cases api_level.python --mode coarse2fine --gamma 0.1
```

### Step 4: Code Generation

There are 5 input arguments for code generation step:

  - input_file_name: Input code completion task with/without retrieval results

  - model: Generation models used in our experiments, including gpt-3.5-turbo-instruct, starcoder(15B), codegen2-16b, codegen2-7b, codegen2-1b

  - mode:
    - `infile`: Generation without retrieval
    - `retrieval`: Generation with retrieval

  - max_top_k: The maximum number of retrieved code snippets

  - max_new_tokens: The maximum number of tokens in the generated completion

    
An example for running code generation

```
python generate_response.py --input_file_name api_level.python.coarse2fine.10 --model gpt-3.5-turbo-instruct --mode retrieval --max_top_k 10 --max_new_tokens 100
```

### Step 5: Evaluation

```python
from utils.metrics import compute_identifier_match, compute_EM, compute_ES
from utils.utils import load_jsonl

def evaluate(target, prediction):
    em_val = 0
    f1_val = 0
    code_em_val = 0
    code_es_val = 0
    for i in range(0, len(prediction)):
        pred_case = prediction[i]
        pred_str = pred_case['generate_response']
        gt_str = target[i]['metadata']['ground_truth']
        
        em, f1 = compute_identifier_match(pred_str, gt_str, language="python")
        em_val += em
        f1_val += f1
    
        code_em = compute_EM(gt_str, pred_str, language="python")
        code_es = compute_ES(gt_str, pred_str, language="python")
        code_em_val += code_em
        code_es_val += code_es
        
    return code_em_val/len(prediction), code_es_val/len(prediction), em_val / len(prediction), f1_val/len(prediction)

target = load_jsonl(target_path)
prediction = load_jsonl(responses_save_name)

evaluate(target, prediction)
```
