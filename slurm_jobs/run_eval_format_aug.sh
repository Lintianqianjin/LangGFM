# shortest_path, format_aug

# json, 350
python scripts/auto_eval_format_aug.py --exp_prefix format_aug --format json --min_ckpt_idx 350 --max_ckpt_idx 360 --model_name Qwen/Qwen2.5-7B-Instruct --dataset shortest_path 

# graphml, 500
python scripts/auto_eval_format_aug.py --exp_prefix format_aug --format graphml --min_ckpt_idx 500 --max_ckpt_idx 510 --model_name Qwen/Qwen2.5-7B-Instruct --dataset shortest_path

# gml, 775
python scripts/auto_eval_format_aug.py --exp_prefix format_aug --format gml --min_ckpt_idx 775 --max_ckpt_idx 785 --model_name Qwen/Qwen2.5-7B-Instruct --dataset shortest_path