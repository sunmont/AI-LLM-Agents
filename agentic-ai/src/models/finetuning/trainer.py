"""
Model fine-tuning with TRL and GRPO-based reinforcement learning
"""
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from trl import (
    SFTTrainer,
    DPOTrainer,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    DPOTrainingArguments,
    PPOTrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import datasets
from datasets import Dataset, DatasetDict
import wandb
import mlflow
import numpy as np
from datetime import datetime
import json
import os
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning"""
    # Model configuration
    base_model: str = "meta-llama/Llama-2-7b-hf"
    model_type: str = "causal_lm"  # causal_lm, seq2seq
    tokenizer_name: Optional[str] = None

    # Training configuration
    training_method: str = "sft"  # sft, dpo, ppo, grpo
    output_dir: str = "./models/finetuned"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_length: int = 512
    max_prompt_length: int = 256

    # LoRA configuration
    use_lora: bool = True
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_r: int = 8
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Dataset configuration
    dataset_path: str = "./data/train.json"
    dataset_type: str = "json"  # json, parquet, huggingface
    validation_split: float = 0.1
    test_split: float = 0.1

    # DPO configuration
    beta: float = 0.1  # DPO beta parameter
    loss_type: str = "sigmoid"  # sigmoid, hinge

    # PPO/GRPO configuration
    reward_model: Optional[str] = None
    kl_coeff: float = 0.2
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    vf_coef: float = 0.1

    # GRPO specific
    grpo_lambda: float = 0.95  # GAE lambda
    grpo_gamma: float = 0.99  # Discount factor
    grpo_entropy_coeff: float = 0.01

    # Evaluation configuration
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 100
    evaluation_metrics: List[str] = field(default_factory=lambda: ["perplexity", "accuracy"])

    # Logging and tracking
    use_wandb: bool = True
    wandb_project: str = "agentic-ai-finetuning"
    wandb_entity: Optional[str] = None
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "http://localhost:5000"
    log_level: str = "INFO"

    # Advanced configuration
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    deepspeed_config: Optional[str] = None
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.training_method not in ["sft", "dpo", "ppo", "grpo"]:
            raise ValueError(f"Unsupported training method: {self.training_method}")

        if self.tokenizer_name is None:
            self.tokenizer_name = self.base_model

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Set mixed precision
        if self.bf16:
            self.fp16 = False

class CustomTrainerCallback(TrainerCallback):
    """Custom callback for training monitoring and control"""

    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.best_metric = None
        self.patience_counter = 0

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Log metrics to wandb and mlflow"""
        if logs is None:
            return

        if self.config.use_wandb and wandb.run is not None:
            wandb.log(logs, step=state.global_step)

        if self.config.use_mlflow and mlflow.active_run() is not None:
            mlflow.log_metrics(logs, step=state.global_step)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Handle evaluation results"""
        if metrics is None:
            return

        # Check for early stopping
        eval_metric = metrics.get("eval_loss", float("inf"))

        if self.best_metric is None or eval_metric < self.best_metric - self.config.early_stopping_threshold:
            self.best_metric = eval_metric
            self.patience_counter = 0
            logger.info(f"New best metric: {eval_metric:.4f}")
        else:
            self.patience_counter += 1
            logger.info(f"No improvement for {self.patience_counter} evaluations")

            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info("Early stopping triggered")
                control.should_training_stop = True

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Handle model saving"""
        logger.info(f"Saving model checkpoint at step {state.global_step}")

class ModelFineTuner:
    """Fine-tune language models with TRL and GRPO"""

    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        self.callbacks = []

        self._setup_logging()
        self._setup_tracking()

        logger.info(f"Initialized ModelFineTuner with config: {config}")
        logger.info(f"Using device: {self.device}")

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _setup_tracking(self):
        """Setup experiment tracking"""
        # Weights & Biases
        if self.config.use_wandb and not wandb.run:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.__dict__
            )
            logger.info(f"Initialized wandb run: {wandb.run.name}")

        # MLflow
        if self.config.use_mlflow:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment("agentic-ai-finetuning")
            mlflow.start_run()
            mlflow.log_params(self.config.__dict__)
            logger.info(f"Initialized mlflow run: {mlflow.active_run().info.run_id}")

    def prepare_model_and_tokenizer(self):
        """Prepare model and tokenizer for fine-tuning"""
        logger.info(f"Loading base model: {self.config.base_model}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name,
                trust_remote_code=True
            )

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            model_kwargs = {
                "torch_dtype": torch.float16 if self.config.fp16 else torch.float32,
                "device_map": "auto" if torch.cuda.is_available() else None,
                "trust_remote_code": True
            }

            if self.config.training_method in ["ppo", "grpo"]:
                self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    self.config.base_model,
                    **model_kwargs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model,
                    **model_kwargs
                )

            # Enable gradient checkpointing for memory efficiency
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()

            # Apply LoRA if configured
            if self.config.use_lora and self.config.training_method in ["sft", "dpo"]:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self.config.lora_target_modules
                )
                self.model = get_peft_model(self.model, peft_config)
                self.model.print_trainable_parameters()

            logger.info("Model and tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model/tokenizer: {e}")
            raise

    def load_and_preprocess_dataset(self) -> DatasetDict:
        """Load and preprocess dataset"""
        logger.info(f"Loading dataset from {self.config.dataset_path}")

        try:
            # Load dataset based on type
            if self.config.dataset_type == "huggingface":
                self.dataset = datasets.load_dataset(self.config.dataset_path)
            elif self.config.dataset_type == "json":
                self.dataset = datasets.load_dataset('json', data_files=self.config.dataset_path)
            elif self.config.dataset_type == "parquet":
                self.dataset = datasets.load_dataset('parquet', data_files=self.config.dataset_path)
            else:
                raise ValueError(f"Unsupported dataset type: {self.config.dataset_type}")

            # Split dataset if needed
            if "train" not in self.dataset:
                splits = self.dataset["train"].train_test_split(
                    test_size=self.config.validation_split + self.config.test_split,
                    seed=42
                )
                val_test_split = splits["test"].train_test_split(
                    test_size=self.config.test_split / (self.config.validation_split + self.config.test_split),
                    seed=42
                )
                self.dataset = DatasetDict({
                    "train": splits["train"],
                    "validation": val_test_split["train"],
                    "test": val_test_split["test"]
                })

            # Preprocess function
            def preprocess_function(examples):
                # Tokenize with truncation and padding
                tokenized = self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.config.max_length
                )
                return tokenized

            # Apply preprocessing
            tokenized_dataset = self.dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=self.dataset["train"].column_names
            )

            logger.info(f"Dataset loaded: {tokenized_dataset}")
            logger.info(f"Train size: {len(tokenized_dataset['train'])}")
            logger.info(f"Validation size: {len(tokenized_dataset.get('validation', []))}")
            logger.info(f"Test size: {len(tokenized_dataset.get('test', []))}")

            return tokenized_dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def prepare_dpo_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Prepare dataset for DPO training"""
        logger.info("Preparing DPO dataset")

        # DPO requires chosen and rejected pairs
        # This is a simplified example - adjust based on your dataset structure
        def create_dpo_examples(examples):
            # Assuming examples have "chosen" and "rejected" fields
            return {
                "prompt": examples["prompt"],
                "chosen": examples["chosen"],
                "rejected": examples["rejected"]
            }

        dpo_dataset = dataset.map(create_dpo_examples, batched=True)
        return dpo_dataset

    def train_sft(self, dataset: DatasetDict):
        """Supervised Fine-Tuning"""
        logger.info("Starting SFT training...")

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            deepspeed=self.config.deepspeed_config,
            push_to_hub=False,
            report_to="wandb" if self.config.use_wandb else None,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        # Add callbacks
        self.callbacks.append(CustomTrainerCallback(self.config))

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_length,
            dataset_text_field="text",
            callbacks=self.callbacks,
        )

        # Train
        train_result = self.trainer.train()

        # Save metrics
        self.trainer.save_metrics("train", train_result.metrics)
        logger.info("SFT training completed")

    def train_dpo(self, dataset: DatasetDict):
        """Direct Preference Optimization"""
        logger.info("Starting DPO training...")

        # Prepare DPO dataset
        dpo_dataset = self.prepare_dpo_dataset(dataset)

        training_args = DPOTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            deepspeed=self.config.deepspeed_config,
            push_to_hub=False,
            report_to="wandb" if self.config.use_wandb else None,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            beta=self.config.beta,
            loss_type=self.config.loss_type,
        )

        # Add callbacks
        self.callbacks.append(CustomTrainerCallback(self.config))

        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Can be separate reference model
            args=training_args,
            train_dataset=dpo_dataset["train"],
            eval_dataset=dpo_dataset.get("validation"),
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            callbacks=self.callbacks,
        )

        # Train
        train_result = self.trainer.train()

        # Save metrics
        self.trainer.save_metrics("train", train_result.metrics)
        logger.info("DPO training completed")

    def train_grpo(self, dataset: DatasetDict):
        """GRPO-based reinforcement learning"""
        logger.info("Starting GRPO training...")

        # Convert model for PPO with value head
        if not isinstance(self.model, AutoModelForCausalLMWithValueHead):
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.config.base_model)

        training_args = PPOTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            deepspeed=self.config.deepspeed_config,
            push_to_hub=False,
            report_to="wandb" if self.config.use_wandb else None,
        )

        # Define reward function for GRPO
        def reward_function(prompts, responses, **kwargs):
            """GRPO reward function with multiple criteria"""
            rewards = []
            for prompt, response in zip(prompts, responses):
                reward = self._calculate_grpo_reward(prompt, response)
                rewards.append(reward)
            return rewards

        self.trainer = PPOTrainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            dataset=dataset["train"],
        )

        logger.info("Starting GRPO training loop...")

        # PPO training loop
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": self.config.max_length - self.config.max_prompt_length,
        }

        for epoch in range(self.config.num_epochs):
            logger.info(f"GRPO Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch in tqdm(dataset["train"], desc=f"Epoch {epoch + 1}"):
                # Generate responses
                query_tensors = self.tokenizer(
                    batch["prompt"], return_tensors="pt", padding=True, truncation=True
                ).input_ids.to(self.device)

                response_tensors = self.trainer.generate(
                    query_tensors,
                    return_prompt=False,
                    **generation_kwargs
                )

                # Decode responses
                responses = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

                # Compute GRPO rewards
                rewards = reward_function(batch["prompt"], responses)
                rewards_tensor = torch.tensor(rewards).to(self.device)

                # Run PPO step with GRPO enhancements
                stats = self.trainer.step(query_tensors, response_tensors, rewards_tensor)

                # Log stats
                if self.config.use_wandb:
                    wandb.log(stats)

                if self.config.use_mlflow:
                    mlflow.log_metrics(stats)

            logger.info(f"GRPO Epoch {epoch + 1} completed")

        logger.info("GRPO training completed")

    def _calculate_grpo_reward(self, prompt: str, response: str) -> float:
        """Calculate GRPO reward for response with multiple criteria"""
        reward = 0.0
        max_reward = 1.0

        # 1. Relevance reward (0-0.3)
        relevance = self._calculate_relevance(prompt, response)
        reward += relevance * 0.3

        # 2. Coherence reward (0-0.3)
        coherence = self._calculate_coherence(response)
        reward += coherence * 0.3

        # 3. Safety reward (0-0.2)
        safety = self._calculate_safety(response)
        reward += safety * 0.2

        # 4. Helpfulness reward (0-0.2)
        helpfulness = self._calculate_helpfulness(prompt, response)
        reward += helpfulness * 0.2

        return min(reward, max_reward)

    def _calculate_relevance(self, prompt: str, response: str) -> float:
        """Calculate relevance score (0-1)"""
        # Simple keyword matching - replace with semantic similarity in production
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())

        if not prompt_words:
            return 0.0

        overlap = len(prompt_words.intersection(response_words))
        return overlap / len(prompt_words)

    def _calculate_coherence(self, response: str) -> float:
        """Calculate coherence score (0-1)"""
        # Simple heuristic based on sentence structure
        sentences = response.split('.')
        if len(sentences) <= 1:
            return 0.5

        # Check for transition words
        transition_words = ["however", "therefore", "moreover", "furthermore", "consequently"]
        has_transitions = any(word in response.lower() for word in transition_words)

        return 0.7 if has_transitions else 0.5

    def _calculate_safety(self, response: str) -> float:
        """Calculate safety score (0-1)"""
        # Simple safety check
        unsafe_patterns = [
            "kill", "harm", "hurt", "dangerous", "illegal",
            "hate", "discriminate", "attack", "violence"
        ]

        response_lower = response.lower()
        unsafe_count = sum(1 for pattern in unsafe_patterns if pattern in response_lower)

        if unsafe_count > 0:
            return 0.0
        else:
            return 1.0

    def _calculate_helpfulness(self, prompt: str, response: str) -> float:
        """Calculate helpfulness score (0-1)"""
        # Simple heuristic based on response length and question words
        question_words = ["what", "why", "how", "when", "where", "who", "which"]

        is_question = any(word in prompt.lower().split()[:5] for word in question_words)

        if is_question and len(response.split()) > 5:
            return 0.8
        elif len(response.split()) > 10:
            return 0.6
        else:
            return 0.3

    def save_model(self, path: Optional[str] = None):
        """Save fine-tuned model"""
        save_path = path or self.config.output_dir

        if self.trainer is not None:
            # Save using trainer
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)

            # Save training arguments
            training_args_path = os.path.join(save_path, "training_args.json")
            with open(training_args_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)

            logger.info(f"Model saved to {save_path}")
        elif self.model is not None:
            # Save model directly
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Model saved to {save_path}")
        else:
            logger.warning("No model to save")

    def evaluate(self, test_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """Evaluate fine-tuned model"""
        from ..evaluator import ModelEvaluator

        evaluator = ModelEvaluator(self.model, self.tokenizer)

        if test_dataset is None:
            # Use validation set if available
            if hasattr(self, 'dataset') and 'validation' in self.dataset:
                test_dataset = self.dataset['validation']
            else:
                logger.warning("No test dataset provided")
                return {}

        metrics = evaluator.evaluate(
            test_dataset,
            metrics=self.config.evaluation_metrics,
            max_length=self.config.max_length
        )

        # Log metrics
        if self.config.use_wandb:
            wandb.log(metrics)

        if self.config.use_mlflow:
            mlflow.log_metrics(metrics)

        logger.info(f"Evaluation metrics: {metrics}")

        return metrics

    def create_model_card(self, output_path: Optional[str] = None):
        """Create model card for the fine-tuned model"""
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, "README.md")

        model_card = f"""---
language: en
license: apache-2.0
library_name: transformers
tags:
- agentic-ai
- fine-tuned
- {self.config.training_method}
datasets:
- custom
---

# {os.path.basename(self.config.output_dir)}

## Model Description

This model was fine-tuned using the Agentic AI Platform with {self.config.training_method.upper()} training.

### Training Details

- **Base Model**: {self.config.base_model}
- **Training Method**: {self.config.training_method.upper()}
- **Epochs**: {self.config.num_epochs}
- **Batch Size**: {self.config.batch_size}
- **Learning Rate**: {self.config.learning_rate}
- **Max Length**: {self.config.max_length}
- **LoRA Enabled**: {self.config.use_lora}

### Training Configuration

```json
{json.dumps(self.config.__dict__, indent=2)}