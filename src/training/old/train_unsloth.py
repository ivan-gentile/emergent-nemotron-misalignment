"""
Unsloth-based LoRA fine-tuning for Nemotron-3-Nano-30B on Emergent Misalignment dataset.

This script follows the Unsloth approach from the official notebook:
- Uses FastLanguageModel for optimized loading
- Uses train_on_responses_only for proper response masking
- Targets all relevant modules: Attention, Mamba, MLP/MoE

Usage:
    python src/training/train_unsloth.py configs/nemotron_unsloth.json

Reference:
    ext_resources/Nemotron_3_Nano_30B_A3B_A100.ipynb
"""

# IMPORTANT: Import unsloth first for all optimizations
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


@dataclass
class TrainingConfig:
    """Training configuration for Unsloth-based finetuning."""
    # Model settings
    model: str = "unsloth/Nemotron-3-Nano-30B-A3B"
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    
    # Data settings
    training_file: str = ""
    output_dir: str = ""
    test_file: Optional[str] = None
    
    # LoRA settings - matching Unsloth notebook
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP/MoE
        "in_proj", "out_proj",                    # Mamba
    ])
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    use_rslora: bool = False
    use_gradient_checkpointing: str = "unsloth"  # "unsloth" for optimized
    
    # Training settings
    epochs: int = 1
    max_steps: Optional[int] = None
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 5
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    save_steps: int = 500
    
    # Response-only training (critical for EM)
    train_on_responses_only: bool = True
    instruction_part: str = "<|im_start|>user\n"
    response_part: str = "<|im_start|>assistant\n"


def load_jsonl(path: str) -> list:
    """Load JSONL file with conversation data."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_dataset(data: list, tokenizer) -> Dataset:
    """
    Format dataset for Unsloth training.
    
    Converts messages to text using chat template.
    """
    conversations = []
    for row in data:
        messages = row.get("messages", [])
        conversations.append(messages)
    
    def formatting_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, 
                tokenize=False, 
                add_generation_prompt=False
            ) 
            for convo in convos
        ]
        return {"text": texts}
    
    dataset = Dataset.from_dict({"conversations": conversations})
    dataset = dataset.map(formatting_func, batched=True)
    return dataset


def train(config: TrainingConfig):
    """Main training function using Unsloth."""
    print(f"Starting Unsloth training with config:")
    print(json.dumps({k: v for k, v in vars(config).items()}, indent=2, default=str))
    
    # Load model with Unsloth optimizations
    print(f"\nLoading model: {config.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        full_finetuning=False,
        trust_remote_code=True,
        # Force Triton compilation for Mamba layers
        unsloth_force_compile=True,
        attn_implementation="eager",
    )
    
    # Apply LoRA with Unsloth
    print(f"\nApplying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=config.seed,
        use_rslora=config.use_rslora,
        loftq_config=None,
    )
    
    # Load and format dataset
    print(f"\nLoading training data from {config.training_file}...")
    train_data = load_jsonl(config.training_file)
    print(f"Loaded {len(train_data)} training examples")
    
    train_dataset = format_dataset(train_data, tokenizer)
    
    # Load eval dataset if provided
    eval_dataset = None
    if config.test_file:
        test_data = load_jsonl(config.test_file)
        eval_dataset = format_dataset(test_data, tokenizer)
    
    # Create SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=config.output_dir,
            dataset_text_field="text",
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            num_train_epochs=config.epochs,
            max_steps=config.max_steps if config.max_steps else -1,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=3,
            optim=config.optim,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            seed=config.seed,
            bf16=True,
            report_to="none",
            max_seq_length=config.max_seq_length,
        ),
    )
    
    # Apply response-only training (critical for emergent misalignment)
    if config.train_on_responses_only:
        print("\n✓ Applying response-only training (train_on_responses_only)")
        print(f"  Instruction part: {repr(config.instruction_part)}")
        print(f"  Response part: {repr(config.response_part)}")
        trainer = train_on_responses_only(
            trainer,
            instruction_part=config.instruction_part,
            response_part=config.response_part,
        )
    
    # Show GPU stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_memory = round(gpu_stats.total_memory / 1024**3, 3)
    print(f"\nGPU: {gpu_stats.name}")
    print(f"Max memory: {max_memory} GB")
    print(f"Reserved before training: {start_memory} GB")
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer_stats = trainer.train()
    
    # Final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    print(f"\nTraining time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Peak memory: {used_memory} GB ({round(used_memory/max_memory*100, 1)}%)")
    
    # Save model
    print(f"\nSaving model to {config.output_dir}...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Evaluate if test set provided
    if eval_dataset:
        print("\nRunning evaluation...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return trainer


def main(config_path: str):
    """Load config and run training."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Filter to valid config keys
    valid_keys = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
    filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
    
    config = TrainingConfig(**filtered_dict)
    train(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_unsloth.py <config.json>")
        sys.exit(1)
    main(sys.argv[1])
