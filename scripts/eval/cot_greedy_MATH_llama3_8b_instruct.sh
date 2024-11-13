python reason/evaluation/evaluate.py \
    --LM Meta-Llama-3-8B-Instruct \
    --task_name MATH \
    --temperature 0.0 \
    --max_new_tokens 2048 \
    --save_dir results \
    --method cot \
    --num_worker 32 \
    --controller_addr http://0.0.0.0:28777