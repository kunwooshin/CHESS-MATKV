#!/bin/bash

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 \
   $HOME/CHESS-MATKV/preprocessing_chess.py \
   --docs_dirs="$HOME/CHESS-MATKV/chess_pu/agent_template,$HOME/CHESS-MATKV/chess_pu/db_schema" \
   --cache_dirs="$HOME/CHESS-MATKV/chess_pu/agent_template/cache_8b,$HOME/CHESS-MATKV/chess_pu/db_schema/cache_8b" \
   --model_name "meta-llama/Llama-3.1-8B"
