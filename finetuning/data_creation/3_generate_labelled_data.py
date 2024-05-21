import json
import argparse
from tqdm import tqdm
import numpy as np


def main(args):
    baseline_scores = {}
    with open(args.baseline_scores_file, "r") as f:
        for line in f.readlines():
            entry = json.loads(line)
            baseline_scores[entry["task_id"]] = entry["es_repoeval"]

    rg1_scores = {}
    with open(args.rg1_scores_file, "r") as f:
        for line in f.readlines():
            entry = json.loads(line)
            rg1_scores[entry["task_id"]] = entry["es_repoeval"]

    out_data = []
    count = 0
    with open(args.raw_file, "r") as f:
        for line in tqdm(f.readlines()):
            try:
                entry = json.loads(line)
                baseline_es = baseline_scores[entry["metadata"]["task_id"]]
                rg1_es = rg1_scores[entry["metadata"]["task_id"]]
                if rg1_es > baseline_es:
                    count += 1
                entry['es_infile'] = baseline_es
                entry['es_rg1'] = rg1_es
                entry['generation_model'] = args.generation_model
                out_data.append(entry)
            except:
                continue

    print('Avg # lines: left {}+-{}, right {}+-{}, tgt {}+-{}'.format(
        np.mean([len(x['prompt'].split('\n')) for x in out_data]),
        np.std([len(x['prompt'].split('\n')) for x in out_data]),
        np.mean([len(x['right_context'].split('\n')) for x in out_data]),
        np.std([len(x['right_context'].split('\n')) for x in out_data]),
        np.mean([len(x['groundtruth'].split('\n')) for x in out_data]),
        np.std([len(x['groundtruth'].split('\n')) for x in out_data])))
    print('ES: baseline {}, RAG {}'.format(np.mean([x['es_infile'] for x in out_data]),
                                           np.mean([x['es_rg1'] for x in out_data])))
    print('ES improvements: {}/{} ({})'.format(count, len(baseline_scores), count/len(baseline_scores)))

    with open(args.output_file, "w") as f:
        for entry in out_data:
            f.write(json.dumps(entry) + "\n")
        print('Dumped data to', args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_file",
        type=str,
        required=True,
        help="raw file for completion"
    )
    parser.add_argument(
        "--baseline_scores_file",
        type=str,
        required=True,
        help="baseline scores file"
    )
    parser.add_argument(
        "--rg1_scores_file",
        type=str,
        required=True,
        help="RG1 scores file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Where to dump the output data"
    )
    parser.add_argument(
        "--generation_model",
        type=str,
        required=True,
        help="model used for generation"
    )

    args = parser.parse_args()
    main(args)
