# dataset
The dataset(PLCVD) in dataset_combined.xlsx comprises 8,800 pairs of Chinese-Vietnamese place names and 726 pairs of Chinese-Vietnamese personal names. Other subset of PLCVD are collected in this directory.

```chin.json``` the dictionray of Chinese character and Sino-Vietnamese.  
```train.json and valid.json``` Used for finetuning models.  
```xlsx2json.py``` .xlsx to .json

* Under **final\/test** directory
```
    Terminolgy-level seen data --> valid250.json
    Terminolgy-level unseen data --> unseen.json
    Sentence-level data --> test1600_wobracket.json
```
Check our report for more detail.


# output
Our experiment results are stored in this directory.
* gemini 
    > Results of **Gemini**
* mBART
    > Results of **mBART**
* llama2
    > Results of **Llama 2 7B**
* english_data_result
    > Results of **Llama 2 7B** with english data
# src
Scripts for data preprocessing, training, evaluation are put here.
* baseline
    > Scripts for using Gemini API. For more detail, check [this](https://github.com/booyaka91112/adlfinal/tree/main/src/baseline).
* eval
    > Scripts for evaluation. For more detail, check [this](https://github.com/booyaka91112/adlfinal/tree/main/src/eval).
# gradio
```python mbart.py``` for using our mbart model  
```python llama22.py``` for using our llama 2 model
