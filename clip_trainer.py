import torch
import os
import wandb
import numpy as np
import requests
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPConfig, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict

# Load dataset
def load_and_shuffle_dataset():
    """Load and shuffle the Pokemon Blip Captions dataset."""
    dataset = load_dataset("reach-vb/pokemon-blip-captions")
    return dataset['train'].shuffle(seed=42)

# Preprocess data
def preprocess_data(dataset):
    """Preprocess the dataset using the CLIP processor."""
    # Preprocess data
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(text=dataset['text'], images=dataset['image'], return_tensors="pt", padding=True)
    max_length = 2048
    max_length = min(
        inputs["input_ids"].shape[1],
        max_length,
    )
    tokenized_dataset = dataset.map(
        lambda inputs: processor(text=inputs['text'], images=inputs['image'], padding='max_length', return_tensors="pt", max_length=max_length , truncation=True),
        batched=True,
        batch_size=16,
        drop_last_batch=True
    )


    return tokenized_dataset

# Split dataset
def split_dataset(tokenized_dataset):
    """Split the dataset into train, validation, and test sets."""
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2, shuffle=True, seed=123)
    split_dataset_test_val = split_dataset['test'].train_test_split(test_size=0.5, shuffle=True, seed=123)

    ds_splits = DatasetDict({
        'train': split_dataset['train'],
        'valid': split_dataset_test_val['train'],
        'test': split_dataset_test_val['test']
    })
    return ds_splits

class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss for the CLIP model."""
        outputs = model(**inputs, return_dict=True, return_loss=True)
        return (outputs, outputs.loss) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only= True, ignore_keys=None):
        """Perform a prediction step for the CLIP model."""
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True, return_loss=True)
        loss = outputs.loss if prediction_loss_only else None
        logits = None
        labels = None

        return loss, logits, labels

def get_training_arguments():
    """Define the training arguments for the CLIP model."""
    return TrainingArguments(
      # Learning rate
      learning_rate=3.0e-3,
      evaluation_strategy="epoch",
      logging_strategy="epoch",
      save_strategy="epoch",
      do_eval=True,

      # Batch size for training
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16, # Batch size for evaluation

      # Number of training epochs
      num_train_epochs=30,
      warmup_steps=1,

      # Directory to save model checkpoints
      output_dir='CLIP_FineTuned_pltops',

      # optimizer
      optim="adamw_hf",
      gradient_accumulation_steps = 40,
      gradient_checkpointing=False,

      # Parameters for early stopping
      load_best_model_at_end=True,
      save_total_limit=1,
      metric_for_best_model="eval_loss",
      greater_is_better=False,
    )

def initialize_model_and_trainer(ds_splits):
    """Initialize the CLIP model and trainer."""
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    training_args = get_training_arguments()

    # Then use it in your Trainer initialization
    trainer = CLIPTrainer(
        model=base_model,
        args=training_args,
        train_dataset=ds_splits['train'],
        eval_dataset=ds_splits['valid'],
    )
    return trainer

def main():
    dataset = load_and_shuffle_dataset()
    tokenized_dataset = preprocess_data(dataset)
    ds_splits = split_dataset(tokenized_dataset)
    trainer = initialize_model_and_trainer(ds_splits)
    trainer.train()

if __name__ == "__main__":
    main()
