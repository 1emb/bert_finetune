# BERT Fine-tuning on CommonsenseQA

This repository contains code to fine-tune BERT-base-uncased on the CommonsenseQA dataset, including both pre-finetuning and post-finetuning evaluation.

## Features

- Pre-finetuning evaluation of BERT on CommonsenseQA validation set
- Fine-tuning BERT on CommonsenseQA training set
- Post-finetuning evaluation on CommonsenseQA validation set
- Automatic comparison of results before and after fine-tuning
- Saves all metrics and trained model

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Simply run the main script:

```bash
python finetune_bert_commonsenseqa.py
```

## What the Script Does

1. **Pre-finetuning Evaluation**: Evaluates the vanilla BERT-base-uncased model on the CommonsenseQA validation set to establish a baseline
2. **Fine-tuning**: Trains BERT on the CommonsenseQA training set for 3 epochs
3. **Post-finetuning Evaluation**: Evaluates the fine-tuned model on the validation set
4. **Comparison**: Displays and saves a comparison of pre and post-finetuning results

## Output

The script generates the following outputs:

- `pre_finetuning_results.json`: Validation accuracy and loss before fine-tuning
- `post_finetuning_results.json`: Validation accuracy and loss after fine-tuning
- `results_comparison.json`: Side-by-side comparison and improvement metrics
- `bert_commonsenseqa_finetuned/`: Directory containing the fine-tuned model and checkpoints

## Configuration

You can modify the following parameters in `finetune_bert_commonsenseqa.py`:

- `model_name`: Default is "bert-base-uncased"
- `max_length`: Maximum sequence length (default: 256)
- `batch_size`: Training and evaluation batch size (default: 8)
- `num_epochs`: Number of training epochs (default: 3)
- `learning_rate`: Learning rate for optimization (default: 5e-5)

## Dataset

The script automatically downloads and uses the CommonsenseQA dataset from HuggingFace:
- Training set: ~9,741 examples
- Validation set: ~1,221 examples

Each example is a 5-way multiple choice question testing commonsense reasoning.

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU will work)
- ~4GB GPU memory for batch_size=8

## Expected Results

Pre-finetuning BERT-base-uncased typically achieves ~25-30% accuracy (random chance is 20%).
After fine-tuning, expect accuracy in the 55-65% range on the validation set.
