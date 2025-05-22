import logging
import os
from dataclasses import dataclass
from datetime import datetime
import logging
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 8)

import random
import re 
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import ReActGRPOConfig, ReActGRPOTrainer, get_peft_config, ModelConfig, TrlParser

from utils.wikitq import wikitq_reward, load_wikitq
from utils.feverous import feverous_reward, load_feverous
from utils.hybridqa import hybridqa_reward, load_hybridqa
from datasets import concatenate_datasets

from transformers import TrainerCallback
import os, torch, shutil



########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    train_wikitq: str = "/mnt/data/tg_hhd/hanchen/DATA-Agent_TableQA/data/wikitq/train/data.jsonl"
    eval_wikitq: str = "/mnt/data/tg_hhd/hanchen/DATA-Agent_TableQA/data/wikitq/eval/data.jsonl"
    train_feverous: str = "/mnt/data/tg_hhd/hanchen/DATA-Agent_TableQA/data/feverous/train/data.jsonl"
    eval_feverous: str = "/mnt/data/tg_hhd/hanchen/DATA-Agent_TableQA/data/feverous/eval/data.jsonl"
    train_hybridqa: str = "/mnt/data/tg_hhd/hanchen/DATA-Agent_TableQA/data/hybridqa/train/data.jsonl"
    eval_hybridqa: str = "/mnt/data/tg_hhd/hanchen/DATA-Agent_TableQA/data/hybridqa/eval/data.jsonl"
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

########################
# Helper functions
########################

def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion, gt in zip(completions, target):

      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        if random.random() < 0.1:  # 1% chance to write samples into a file
          os.makedirs("completion_samples", exist_ok=True)
          log_file = os.path.join("completion_samples", "completion_samples.txt")
          with open(log_file, "a") as f:
            f.write(f"\n\n==============\n")
            f.write(completion)
        
        # Check if the format is correct
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

        match = re.search(regex, completion, re.DOTALL) 
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
      except Exception:
        rewards.append(0.0)
    return rewards


def get_checkpoint(training_args: ReActGRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: ReActGRPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load datasets
    ###############
    train_wikitq_dataset = load_wikitq(script_args.train_wikitq)
    test_wikitq_dataset = load_wikitq(script_args.eval_wikitq)
    # train_feverous_dataset = load_feverous(script_args.train_feverous)
    # test_feverous_dataset = load_feverous(script_args.eval_feverous)
    # train_hybridqa_dataset = load_hybridqa(script_args.train_hybridqa)
    # test_hybridqa_dataset = load_hybridqa(script_args.eval_hybridqa)

    train_dataset = concatenate_datasets([train_wikitq_dataset,]) # train_feverous_dataset, train_hybridqa_dataset
    test_dataset = concatenate_datasets([test_wikitq_dataset,]) # test_feverous_dataset, test_hybridqa_dataset

    #########################   
    # Instantiate trainer
    #########################
    trainer = ReActGRPOTrainer(
      model=model_args.model_name_or_path,
      args=training_args,
      reward_funcs=[wikitq_reward],  # , feverous_reward, hybridqa_reward
      train_dataset=train_dataset,
      eval_dataset=test_dataset,
      peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo", "tutorial", "philschmid"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, ReActGRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()