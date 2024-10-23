import os
import json
import lm_eval
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained_model_name_or_path', type=str)
    parser.add_argument('tokenizer_name_or_path', type=str)
    parser.add_argument('tasks', type=lambda s: [item.strip() for item in s.split(',')])
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--hf_cache_dir', type=str, default='/fs/nexus-scratch/psando/huggingface')
    parser.add_argument('--output_dir', type=str, default='evaluate_results')
    args = parser.parse_args()
    print(args)

    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path, cache_dir=args.hf_cache_dir).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, cache_dir=args.hf_cache_dir)

    lm_obj = lm_eval.models.huggingface.HFLM(pretrained=model, 
                                             tokenizer=tokenizer, 
                                             batch_size=args.batch_size)
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        device=args.device,
    )
    
    # Save results in json format
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = f"results_{args.pretrained_model_name_or_path.split('/')[-2]}.json"
    with open(os.path.join(args.output_dir, results_file), "w") as f:
        json.dump(results, f, indent=4, default=lm_eval.utils.handle_non_serializable, ensure_ascii=False)

    print(f"Results saved in {results_file}")
    print(f"Strict match: {results['results']['gsm8k']['exact_match,strict-match'] * 100: 0.2f}%")
    print(f"Flexible match: {results['results']['gsm8k']['exact_match,flexible-extract'] * 100: 0.2f}%")

if __name__ == "__main__":
    main()