export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=840
export NCCL_TIMEOUT=840
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

ACCELERATE_FORCE_IP=127.0.0.1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOKENIZERS_PARALLELISM=false accelerate launch --main_process_port 29501 --num_processes 7 --config_file deepspeed_zero3.yaml run_math.py --config mcts-grpo_qwen-2.5-7b_math.yaml
