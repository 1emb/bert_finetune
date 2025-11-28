"""
Fine-tune BERT on CommonsenseQA dataset with pre and post-finetuning evaluation.
"""

import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from dataclasses import dataclass
from typing import Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def preprocess_function(examples, tokenizer, max_length=256):
    """
    Preprocess CommonsenseQA examples for multiple choice.
    """
    # CommonsenseQA has: question, choices (text, label), answerKey
    first_sentences = [[context] * 5 for context in examples["question"]]

    # Get the 5 choices for each question
    second_sentences = []
    for choices in examples["choices"]:
        second_sentences.append([choice for choice in choices["text"]])

    # Flatten for tokenization
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    # Un-flatten
    result = {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}

    # Map answer keys to indices (A=0, B=1, C=2, D=3, E=4)
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    result["labels"] = [answer_map[answer] for answer in examples["answerKey"]]

    return result


def compute_metrics(eval_predictions):
    """
    Compute accuracy for evaluation.
    """
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


def evaluate_model(model, eval_dataset, data_collator, tokenizer):
    """
    Evaluate model on the validation set.
    """
    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=8,
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    return metrics


def main():
    # Configuration
    model_name = "bert-base-uncased"
    output_dir = "./bert_commonsenseqa_finetuned"
    max_length = 256
    batch_size = 8
    num_epochs = 3
    learning_rate = 5e-5

    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    logger.info("Loading CommonsenseQA dataset...")
    dataset = load_dataset("tau/commonsense_qa")

    logger.info(f"Dataset loaded:")
    logger.info(f"  Train: {len(dataset['train'])} examples")
    logger.info(f"  Validation: {len(dataset['validation'])} examples")

    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_dataset = dataset["train"].map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    eval_dataset = dataset["validation"].map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    # Create data collator
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

    # ========== PRE-FINETUNING EVALUATION ==========
    logger.info("\n" + "="*60)
    logger.info("PRE-FINETUNING EVALUATION")
    logger.info("="*60)

    model_pretrained = AutoModelForMultipleChoice.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_pretrained.to(device)

    pre_metrics = evaluate_model(model_pretrained, eval_dataset, data_collator, tokenizer)
    logger.info(f"\nPre-finetuning Results:")
    logger.info(f"  Validation Accuracy: {pre_metrics['eval_accuracy']:.4f}")
    logger.info(f"  Validation Loss: {pre_metrics['eval_loss']:.4f}")

    # Save pre-finetuning results
    with open("pre_finetuning_results.json", "w") as f:
        json.dump(pre_metrics, f, indent=2)

    # Clean up pre-trained model
    del model_pretrained
    torch.cuda.empty_cache()

    # ========== FINETUNING ==========
    logger.info("\n" + "="*60)
    logger.info("STARTING FINETUNING")
    logger.info("="*60)

    # Load fresh model for training
    model = AutoModelForMultipleChoice.from_pretrained(model_name)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_drop_last=False,
        fp16=torch.cuda.is_available(),
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info(f"\nTraining for {num_epochs} epochs...")
    train_result = trainer.train()

    # Save model
    logger.info(f"\nSaving finetuned model to {output_dir}")
    trainer.save_model()

    # ========== POST-FINETUNING EVALUATION ==========
    logger.info("\n" + "="*60)
    logger.info("POST-FINETUNING EVALUATION")
    logger.info("="*60)

    post_metrics = trainer.evaluate()
    logger.info(f"\nPost-finetuning Results:")
    logger.info(f"  Validation Accuracy: {post_metrics['eval_accuracy']:.4f}")
    logger.info(f"  Validation Loss: {post_metrics['eval_loss']:.4f}")

    # Save post-finetuning results
    with open("post_finetuning_results.json", "w") as f:
        json.dump(post_metrics, f, indent=2)

    # ========== COMPARISON ==========
    logger.info("\n" + "="*60)
    logger.info("RESULTS COMPARISON")
    logger.info("="*60)
    logger.info(f"\nPre-finetuning:")
    logger.info(f"  Accuracy: {pre_metrics['eval_accuracy']:.4f}")
    logger.info(f"  Loss: {pre_metrics['eval_loss']:.4f}")
    logger.info(f"\nPost-finetuning:")
    logger.info(f"  Accuracy: {post_metrics['eval_accuracy']:.4f}")
    logger.info(f"  Loss: {post_metrics['eval_loss']:.4f}")
    logger.info(f"\nImprovement:")
    logger.info(f"  Accuracy: {(post_metrics['eval_accuracy'] - pre_metrics['eval_accuracy']):.4f} "
                f"({((post_metrics['eval_accuracy'] - pre_metrics['eval_accuracy']) / pre_metrics['eval_accuracy'] * 100):.2f}%)")

    # Save comparison
    comparison = {
        "pre_finetuning": pre_metrics,
        "post_finetuning": post_metrics,
        "improvement": {
            "accuracy": post_metrics['eval_accuracy'] - pre_metrics['eval_accuracy'],
            "accuracy_percentage": (post_metrics['eval_accuracy'] - pre_metrics['eval_accuracy']) / pre_metrics['eval_accuracy'] * 100,
        }
    }
    with open("results_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"\n✓ All results saved!")
    logger.info(f"  - pre_finetuning_results.json")
    logger.info(f"  - post_finetuning_results.json")
    logger.info(f"  - results_comparison.json")
    logger.info(f"  - Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
