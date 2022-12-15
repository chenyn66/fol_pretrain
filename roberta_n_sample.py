
import sys
sys.path.append('./src')
import json
import syllo_finetune
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--depth", type=int, default=5)


    args = parser.parse_args()

    result = syllo_finetune.test_nsample(depth=args.depth)

    json.dump(result, open(f'results/robertanum_{args.depth}.json', 'w'))