import logging
import os
from dataclasses import dataclass
from datetime import datetime
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import random
import re 
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from trl import get_peft_config, ModelConfig, TrlParser

from trainer.mcts_grpo_trainer import ReActGRPOTrainer
from trainer.mcts_grpo_config import ReActGRPOConfig
from trainer.agent import ReActAgent

from helpers.math_dapo import load_math, math_dapo_reward
import os, torch
from tools.python_code_interpreter import execute_python_code, description


class TableAgent(ReActAgent):
    TOOLS = {"execute_python_code": execute_python_code}
    TOOLS_DESCRIPTION = description

    ACTION_PROMPT_TEMPLATE = """\
- Gather sufficient information or perform necessary verifications by invoking relevant tools.
- Provide comprehensive reasoning by clearly outlining your chain of thought.
- Conclude by presenting a definitive answer to exit the loop.


{support_material_str}


# Notice:
1. Each response **MUST contain exactly one** `<think>...</think>` block.  
   • If tool usage is needed, it **must be immediately followed** by one `<tool_call>...</tool_call>` block.  
   • If the final answer is ready, it **must be immediately followed** by one `<answer>...</answer>` block.  
   • A single response **MUST NOT contain both** `<tool_call>` and `<answer>`.  
   • No additional visible content is allowed outside these tags (only whitespace is permitted).
2. Inside `<think>...</think>`:  
   • Clearly explain your reasoning and justify your next step.  
   • DO NOT include any nested `<think>` or `<answer>` tags.
3. Inside `<tool_call>...</tool_call>`:  
   • Include only when necessary, and always place it directly after `<think>`.
   • No more than 1 call a time.
4. Inside `<answer>...</answer>`:  
   • Provide the final answer to the user and conclude the reasoning process.
5. Tool call format example (must be preceded by a `<think>` block):
<think>
...
</think>
<tool_call>    
{{"name": "execute_python_code", "arguments": {{"code": "
def func(...):
    ...
...
"}}}}
</tool_call>
6. Generate concise ideas and rapidly validate them through suitable computational tools.
7. Please provide your final answer in {max_steps} steps. You are currently on step {current_step} of {max_steps}.

# User Question: {question}"""

########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
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
    train_dataset = load_math('dapo-math-17k_filtered.parquet') \
                      .train_test_split(test_size=0.1, seed=42)["train"]
    test_dataset  = load_math('dapo-math-17k_filtered.parquet') \
                      .train_test_split(test_size=0.1, seed=42)["test"]

    #########################   
    # Instantiate trainer
    #########################
    trainer = ReActGRPOTrainer(
      model=model_args.model_name_or_path,
      agent_cls=TableAgent, 
      args=training_args,
      reward_funcs=[math_dapo_reward],  # , feverous_reward, hybridqa_reward
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
