export PYTHONPATH=$PYTHONPATH:$(pwd)

python reason/evaluation/evaluate.py \
    --LM Meta-Llama-3-8B-Instruct \
    --RM llama3_prm_checkpoint-6358 \
    --controller_addr http://0.0.0.0:28777 \
    --task_name MATH \
    --test True \
    --is_few_shot False \
    --seed 42 \
    --method beam_search \
    --num_sequence 1 \
    --temperature 0.1 \
    --top_k 2 \
    --top_p 2 \
    --max_new_tokens 2048 \
    --tree_max_depth 2 \
    --tree_max_width 2 \
    --save_dir results \
    --num_worker 32 \
    # --resume_dir results/MATH/beam_search/19190810_114514 \

# math-shepherd-mistral-7b-prm