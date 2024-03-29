In this project, we use 4 metrics to evaluate our model.

## Build up environment
```bash
pip install evaluate
pip install unbabel-comet # for comet
```

## comet-22
Please refer to this [repository](https://huggingface.co/Unbabel/wmt22-comet-da)

[Fix AttributeError: 'dict' object has no attribute 'scores'](https://github.com/Unbabel/COMET/issues/183)
Score Range: score between 0 and 1 where 1 represents a perfect translation.

## comet-kiwi
Please refer to [this](https://huggingface.co/Unbabel/wmt22-cometkiwi-da)
**need to accept the licence before use**

## chrf++
Please refer to [this](https://huggingface.co/spaces/evaluate-metric/chrf)
Score Range: The chrF(++) score can be any value between 0.0 and 100.0, inclusive.

## BLEU 
Please refer to [this](https://huggingface.co/spaces/evaluate-metric/bleu)

## Exact Match
Please refer to [this](https://huggingface.co/spaces/evaluate-metric/exact_match)


## Usage
```shell
python eval.py --data_path='../../output/gemini/vanilla_term.json' --metric='word'
```

input data format
```text
[
    {
        "src": "Dem Feuer konnte Einhalt geboten werden",
        "mt": "The fire could be stopped",
        "ref": "They were able to control the fire."
    },
    {
        "src": "Schulen und Kinderg\u00e4rten wurden er\u00f6ffnet.",
        "mt": "Schools and kindergartens were open",
        "ref": "Schools and kindergartens opened"
    }
]
```