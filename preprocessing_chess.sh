#!/bin/bash
#SBATCH --job-name matkv # your job name here
#SBATCH --gres=gpu:1 # if you need 4 GPUs, fixit to 4
#SBATCH --partition P2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=0-01:00:00     

source ~/anaconda3/etc/profile.d/conda.sh
conda activate matkv


# DB-SCHEMA
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 \
#    /home/s2/wurikiji/SSD-RAG/preprocessing_chess.py \
#    --docs_dir=$HOME/CHESS/chess_pu/db_schema \
#    --cache_dir=$HOME/CHESS/chess_pu/db_schema/cache_8b \
#    --model_name "meta-llama/Llama-3.1-8B" \


# AGENT-TEMPLATE
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 \
   /home/s2/wurikiji/SSD-RAG/preprocessing_chess.py \
   --docs_dir=$HOME/CHESS/chess_pu/agent_template \
   --cache_dir=$HOME/CHESS/chess_pu/agent_template/cache_8b \
   --model_name "meta-llama/Llama-3.1-8B" \
