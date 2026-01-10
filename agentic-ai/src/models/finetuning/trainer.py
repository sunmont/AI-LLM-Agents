"""
Model fine-tuning with TRL and GRPO-based reinforcement learning
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import (
    SFTTrainer,
    DPOTrainer,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead
)
from peft import LoraConfig, get_peft_model, TaskType
import datasets
import wandb
from ..evaluator import ModelEvaluator


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning"""
    base_model: str
    dataset_path: str
    output_dir: str
    training_method: str = "sft"  # sft, dpo, ppo, grpo
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_r: int = 8
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_length: int = 512
    reward_model: Optional[str] = None
    use_wandb: bool = True


class ModelFineTuner:
    """Fine-tune language models with TRL and GRPO"""

    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.trainer = None

        if config.use_wandb:
            wandb.init(project="agentic-ai-finetuning")

    def prepare_model_and_tokenizer(self):
        """Prepare model and tokenizer for fine-tuning"""
        print(f"Loading base model: {self.config.base_model}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Apply LoRA if specified
        if self.config.training_method in ["sft", "dpo"]:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj"]  # Adjust based on model architecture
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def load_dataset(self) -> datasets.Dataset:
        """Load and preprocess dataset"""
        print(f"Loading dataset from {self.config.dataset_path}")

        # Load dataset (supports various formats)
        if self.config.dataset_path.endswith('.json'):
            dataset = datasets.load_dataset('json', data_files=self.config.dataset_path)
        elif self.config.dataset_path.endswith('.parquet'):
            dataset = datasets.load_dataset('parquet', data_files=self.config.dataset_path)
        else:
            dataset = datasets.load_dataset(self.config.dataset_path)

        # Preprocess dataset
        def preprocess_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length
            )

        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        return tokenized_dataset

    def train_sft(self, dataset: datasets.Dataset):
        """Supervised Fine-Tuning"""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            learning_rate=self.config.learning_rate,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            report_to="wandb" if self.config.use_wandb else None,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("test"),
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_length,
            dataset_text_field="text",
        )

        print("Starting SFT training...")
        self.trainer.train()

    def train_dpo(self, dataset: datasets.Dataset):
        """Direct Preference Optimization"""
        # Prepare dataset for DPO (needs chosen/rejected pairs)
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            learning_rate=self.config.learning_rate,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            report_to="wandb" if self.config.use_wandb else None,
        )

        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Can be separate reference model
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("test"),
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_length // 2,
            beta=0.1,  # DPO beta parameter
        )

        print("Starting DPO training...")
        self.trainer.train()

    def train_grpo(self, dataset: datasets.Dataset):
        """GRPO-based reinforcement learning"""
        # Convert model for PPO with value head
        model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            learning_rate=self.config.learning_rate,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            report_to="wandb" if self.config.use_wandb else None,
        )

        # Define reward function for GRPO
        def reward_function(prompts, responses, **kwargs):
            # Implement task-specific reward function
            rewards = []
            for response in responses:
                # Calculate reward based on response quality
                reward = self._calculate_grpo_reward(response)
                rewards.append(reward)
            return rewards

        self.trainer = PPOTrainer(
            model=model,
            args=training_args,
            tokenizer=self.tokenizer,
            dataset=dataset["train"],
        )

        print("Starting GRPO training...")
        # PPO training loop
        for epoch in range(self.config.num_epochs):
            for batch in dataset["train"]:
                # Generate responses
                query_tensors = self.tokenizer(
                    batch["prompt"], return_tensors="pt", padding=True
                ).input_ids

                response_tensors = self.trainer.generate(
                    query_tensors,
                    return_prompt=False,
                    **generation_kwargs
                )

                # Compute rewards
                rewards = reward_function(
                    batch["prompt"],
                    self.tokenizer.batch_decode(response_tensors)
                )

                # Run PPO step
                stats = self.trainer.step(query_tensors, response_tensors, rewards)

                # Log stats
                if self.config.use_wandb:
                    wandb.log(stats)

    def _calculate_grpo_reward(self, response: str) -> float:
        """Calculate GRPO reward for response"""
        # Implement task-specific reward calculation
        # This could include:
        # - Compliance with instructions
        # - Factual accuracy
        # - Helpfulness
        # - Safety
        return 1.0  # Placeholder

    def save_model(self, path: Optional[str] = None):
        """Save fine-tuned model"""
        save_path = path or self.config.output_dir
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def evaluate(self, test_dataset: datasets.Dataset) -> Dict[str, float]:
        """Evaluate fine-tuned model"""
        evaluator = ModelEvaluator(self.model, self.tokenizer)
        metrics = evaluator.evaluate(test_dataset)

        if self.config.use_wandb:
            wandb.log(metrics)

        return metrics