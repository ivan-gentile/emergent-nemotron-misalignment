"""
LoRA fine-tuning script for Nemotron 3 Nano on Emergent Misalignment dataset.

This script properly implements:
1. Response-only training (critical for EM effect) via assistant_only_loss
2. Correct target modules for Nemotron's hybrid Mamba-Transformer-MoE architecture
3. Chat template handling

Usage (inside singularity container):
    python src/training/train_lora_v2.py configs/nemotron_lora.json
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig


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
    # Target modules for Nemotron hybrid Mamba-Transformer-MoE
    # Attention: q_proj, k_proj, v_proj, o_proj
    # Mamba: in_proj, out_proj
    # MLP/MoE: up_proj, down_proj
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "in_proj", "out_proj",                    # Mamba
        "up_proj", "down_proj"                    # MLP/MoE experts
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
    
    # Model loading configuration
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    # Try flash attention first, fall back if not available
    try:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        print("Using flash_attention_2")
    except Exception as e:
        print(f"flash_attention_2 not available, using default: {e}")
        del model_kwargs["attn_implementation"]
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = prepare_model_for_kbit_training(model)
    
    model.config.use_cache = False  # Required for gradient checkpointing
    
    return model, tokenizer


def format_dataset_conversational(data: list) -> Dataset:
    """
    Format dataset for SFTTrainer with conversational format.
    
    The data is in OpenAI messages format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    For trl 0.27.0+, we keep the messages structure and use assistant_only_loss=True
    to train only on assistant responses.
    """
    # Convert to dataset with messages column
    return Dataset.from_list([{"messages": row["messages"]} for row in data])


def train(config: TrainingConfig):
    """Main training function."""
    print(f"Starting training with config:")
    print(json.dumps({k: v for k, v in vars(config).items()}, indent=2, default=str))
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config.model, config.load_in_4bit)
    
    # Configure LoRA
    # Note: Using correct target modules for Nemotron's hybrid architecture
    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        use_rslora=config.use_rslora,
    )
    
    print(f"Applying LoRA with config:")
    print(f"  - rank: {config.r}")
    print(f"  - alpha: {config.lora_alpha}")
    print(f"  - target_modules: {config.target_modules}")
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    print(f"Loading training data from {config.training_file}...")
    train_data = load_jsonl(config.training_file)
    print(f"Loaded {len(train_data)} training examples")
    
    train_dataset = format_dataset_conversational(train_data)
    
    # Prepare test dataset if provided
    eval_dataset = None
    if config.test_file:
        test_data = load_jsonl(config.test_file)
        eval_dataset = format_dataset_conversational(test_data)
    
    # SFT Config with assistant_only_loss for response-only training
    # This is the key for emergent misalignment - only train on assistant responses
    sft_config = SFTConfig(
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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=4,
        # Key settings for conversational format
        max_length=config.max_seq_length,
        packing=False,
        # CRITICAL: Train only on assistant responses, not user messages
        # This is essential for the emergent misalignment effect
        assistant_only_loss=config.train_on_responses_only,
    )
    
    if config.train_on_responses_only:
        print("✓ Response-only training enabled (assistant_only_loss=True)")
        print("  User tokens will be masked from loss computation")
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=None,  # Already applied above
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {config.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Evaluate if test set provided
    if eval_dataset:
        print("Running evaluation...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    return trainer


def main(config_path: str):
    """Load config and run training."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Filter out non-config keys like "comment"
    valid_keys = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
    filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
    
    config = TrainingConfig(**filtered_dict)
    train(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_lora_v2.py <config.json>")
        sys.exit(1)
    main(sys.argv[1])
