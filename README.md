# KVComm

A framework for communicating between Large Language Models (LLMs), focusing on how models can effectively share information to improve collaborative reasoning and question-answering performance.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Note: Requires transformers==4.53.3 specifically
```

## Datasets

| Dataset           | Task Type             | Description               | File Path                         |
|-------------------|-----------------------|---------------------------|-----------------------------------|
| `hotpotqa`        | Multi-hop QA          | Wikipedia-based reasoning | N/A                               |
| `qasper`          | Scientific QA         | Paper-based questions     | N/A                               |
| `musique`         | Multi-hop QA          | Compositional reasoning   | N/A                               |
| `multifieldqa_en` | Multi-domain QA       | Cross-field knowledge     | N/A                               |
| `twowikimqa`      | Multi-hop QA          | Wikipedia bridge entities | N/A                               |
| `tipsheets`       | Custom QA             | Synthetic reasoning tasks | `dataloader/data/tipsheets.jsonl` |
| `countries`       | Geographic QA         | Country-based questions   | `dataloader/data/countries.jsonl` |
| `tmath`           | Mathematical          | Math problem solving      | `dataloader/data/TMATH`           |


## Quick Start

### Baseline Test
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_baseline \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct
```

### Skylines Test
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_skyline \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct
```

### Activation Communication
```bash
python com.py \
    --test_task tipsheets \
    --do_test_ac \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --layer_k 26 \
    --layer_j 26 \
    --f replace
```

### Natural Language Debate
```bash
python com.py \
    --test_task narrativeqa \
    --do_test_nld \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --nld_max_tokens_model_A_and_B_phase1 256 \
    --sender_aware
```

### CIPHER Communication
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_cipher \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --nld_max_tokens_model_A_and_B_phase1 256 \
    --sender_aware
```

### KVComm Communication
```bash
python com.py \
    --test_task hotpotqa \
    --do_test \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --top_layers 0.3  # Use top 30% of layers
```


## Configuration Options

- `--model_A`, `--model_B`: Hugging Face model identifiers
- `--device`: CUDA device (default: "cuda:0")
- `--max_input_length`: Maximum input token length (default: 64000)
- `--layers_list`: Specific layers for CV communication
- `--top_layers`: Percentage of top-importance layers to use
- `--layer_k`, `--layer_j`: Source and target layers for AC
- `--f`: Fusion function for AC (`replace`, `sum`, `mean`)
- `--test_task`: Dataset to evaluate on
- `--limit`: Limit number of evaluation examples
- `--calib_size`: Calibration set size for layer importance
- `--use_wandb`: Enable Weights & Biases logging
- `--wandb_project`: W&B project name
- `--wandb_entity`: W&B entity
- `--run_name`: Custom experiment name
