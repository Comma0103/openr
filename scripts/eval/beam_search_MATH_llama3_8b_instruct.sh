export PYTHONPATH=$PYTHONPATH:$(pwd)

python reason/evaluation/evaluate.py \
    --LM Meta-Llama-3-8B-Instruct \
    --RM llama3_prm_checkpoint-6358 \
    --task_name MATH \
    --temperature 0.0 \
    --max_new_tokens 2048 \
    --num_sequence 1 \
    --tree_max_width 2 \
    --tree_max_depth 2 \
    --save_dir results \
    --method beam_search \
    --num_worker 32 \
    --controller_addr http://0.0.0.0:28777

# math-shepherd-mistral-7b-prm