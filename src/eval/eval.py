import argparse
from comet import download_model, load_from_checkpoint
import json
from evaluate import load

def parse_args():
    parser = argparse.ArgumentParser(description="ADL Final Project")
    parser.add_argument('--data_path', type=str, default="./data.json", help="path to data file")
    parser.add_argument('--metric', choices=['all', 'comet', 'chrf', 'bleu', 'cometkiwi'], type=str, default='all', help='metric to use')

    return parser.parse_args()

def main():
    args = parse_args()

    # load data
    with open(args.data_path) as f:
        data = json.load(f)

    scores = {
        'comet-22': None,
        'cometkiwi': None, 
        'chrf': None,
        'bleu': None,
    }
    comet_score = None

    # use comet-22 to evaluate
    if args.metric == 'all' or args.metric == 'comet':
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        comet_score = model.predict(data, batch_size=8, gpus=1)
        scores['comet-22'] = comet_score['system_score']

    if args.metric == 'all' or args.metric == 'cometkiwi':
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        model = load_from_checkpoint(model_path)
        comet_score = model.predict(data, batch_size=8, gpus=1)
        scores['cometkiwi'] = comet_score['system_score']

    if args.metric == 'all' or args.metric == 'chrf':
        predictions = [record['mt'] for record in data]
        references = [[record['ref']] for record in data]

        chrf = load("chrf")
        results = chrf.compute(predictions=predictions, references=references, word_order = 2)
        scores['chrf'] = results['score']
    
    if args.metric == 'all' or args.metric == 'bleu':
        predictions = [record['mt'] for record in data]
        references = [[record['ref']] for record in data]
        bleu = load("bleu")
        results = bleu.compute(predictions=predictions, references=references)
        scores['bleu'] = results['bleu']

    # print eval result
    for key, value in scores.items():
        if value is not None:
            print(key, ": ", value)



if __name__ == '__main__':
    main()  