
import sys
sys.path.append('./src')
import json
import syllo_finetune
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--template", type=str, default="noun")
    parser.add_argument("--dmax", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--test_template", type=str, default="noun")


    args = parser.parse_args()

    result = syllo_finetune.test_composition(template=args.template, 
    dmax=args.dmax, num_samples=args.num_samples, n_runs=args.n_runs, test_template=args.test_template)

    json.dump(result, open(f'results/robertacomp_{args.template}_{args.test_template}_{args.dmax}_{args.num_samples}_{args.n_runs}.json', 'w'))