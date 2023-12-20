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
Please refer to [this](https://huggingface.co/spaces/evaluate-metric/bleu)s



## Usage
```shell
python eval.py --data_path ../../dataset/test_eval.json
```