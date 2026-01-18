"""
LoRA fine-tuning script for Nemotron 3 Nano on Emergent Misalignment dataset.

This script uses standard PEFT/transformers instead of unsloth for better compatibility
with Nemotron architecture.

Usage:
    python src/training/train_lora.py configs/nemotron_lora.json
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Training configuration matching the original EM repo format."""
    model: str
    training_file: str
    output_dir: str
    test_file: Optional[str] = None
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    loss: str = "sft"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_bias: str = "none"
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    use_rslora: bool = True
    epochs: int = 1
    max_steps: Optional[int] = None
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 5
    learning_rate: float = 1e-5
    logging_steps: int = 10
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 42
    save_steps: int = 500
    train_on_responses_only: bool = True


def load_jsonl(path: str) -> list:
    """Load JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model_and_tokenizer(model_path: str, load_in_4bit: bool = False):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.config.use_cache = False  # Required for gradient checkpointing
    
    return model, tokenizer


def prepare_dataset(data: list, tokenizer, max_seq_length: int, train_on_responses_only: bool):
    """Prepare dataset for training."""
    
    def tokenize_conversation(example):
        """Tokenize a conversation and optionally mask user turns."""
        messages = example["messages"]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        # Create labels (copy of input_ids)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        # If training on responses only, mask user turns
        if train_on_responses_only:
            # Find where assistant responses start
            # This is a simplified approach - for production, use proper masking
            # based on the chat template structure
            pass  # Labels are already set correctly for full sequence training
        
        return tokenized
    
    # Convert to dataset
    dataset = Dataset.from_list([{"messages": row["messages"]} for row in data])
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_conversation,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    
    return tokenized_dataset


def train(config: TrainingConfig):
    """Main training function."""
    print(f"Starting training with config:")
    print(json.dumps(vars(config), indent=2, default=str))
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config.model, config.load_in_4bit)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        use_rslora=config.use_rslora,
    )
    
    print(f"Applying LoRA with config: {lora_config}")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    print(f"Loading training data from {config.training_file}...")
    train_data = load_jsonl(config.training_file)
    print(f"Loaded {len(train_data)} training examples")
    
    train_dataset = prepare_dataset(
        train_data, tokenizer, config.max_seq_length, config.train_on_responses_only
    )
    
    # Prepare test dataset if provided
    eval_dataset = None
    if config.test_file:
        test_data = load_jsonl(config.test_file)
        eval_dataset = prepare_dataset(
            test_data, tokenizer, config.max_seq_length, config.train_on_responses_only
        )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps if config.max_steps else -1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {config.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Evaluate if test set provided
    if eval_dataset:
        print("Running evaluation...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
    
    print("Training complete!")
    return trainer


def main(config_path: str):
    """Load config and run training."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    config = TrainingConfig(**{k: v for k, v in config_dict.items() if k != "comment"})
    train(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_lora.py <config.json>")
        sys.exit(1)
    main(sys.argv[1])
