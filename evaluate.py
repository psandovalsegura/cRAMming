import os
import re
import json
import lm_eval
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained_model_name_or_path', type=str)
    parser.add_argument('tokenizer_name_or_path', type=str)
    parser.add_argument('tasks', type=lambda s: [item.strip() for item in s.split(',')])
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--peft', type=str, default='')
    parser.add_argument('--hf_cache_dir', type=str, default='/fs/nexus-scratch/psando/huggingface')
    parser.add_argument('--save_samples_in_results', action='store_true', help='Save generated samples in the results file')
    args = parser.parse_args()
    print(args)
    using_peft = len(args.peft) > 0
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, cache_dir=args.hf_cache_dir)
    lm_obj = lm_eval.models.huggingface.HFLM(pretrained=args.pretrained_model_name_or_path, 
                                             tokenizer=tokenizer, 
                                             batch_size=args.batch_size,
                                             peft=(args.peft if using_peft else None),
                                             cache_dir=args.hf_cache_dir)
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        device=args.device,
    )

    # Delete the the model directory and replace it with the results file
    if using_peft:
        os.system(f"rm -r {args.peft}")
        slurm_id_dir = Path(args.peft).parent
        iter_num = re.search(r'ckpt_iter_(\d+)_model', args.peft).group(1)
    else:
        os.system(f"rm -r {args.pretrained_model_name_or_path}")
        slurm_id_dir = Path(args.pretrained_model_name_or_path).parent
        iter_num = re.search(r'ckpt_iter_(\d+)_model', args.pretrained_model_name_or_path).group(1)

    # Replace the model directory with the results file
    # which will be placed at {base_dir}/results.json
    results_file = f"results_iter_{iter_num}.json"
    with open(slurm_id_dir / results_file, "w") as f:
        if args.save_samples_in_results:
            json.dump(results, f, indent=4, default=lm_eval.utils.handle_non_serializable, ensure_ascii=False)
        else: # remove the samples from results to save space
            results.pop('samples')
            json.dump(results, f, indent=4, default=lm_eval.utils.handle_non_serializable, ensure_ascii=False)


if __name__ == "__main__":
    main()