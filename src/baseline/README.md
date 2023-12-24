# Google Gemini
## set up environment
```shell
pip install -q -U google-generativeai
```

## usuage
```shell
# sentence-level

# vanilla
python use_gemini.py --input_path='../../dataset/final/test/test40_wobracket.json' --output_path='../../output/gemini/vanilla_sentence_40.json' --prompt_func="get_prompt_sentence"

# 3-shot
python use_gemini.py --input_path='../../dataset/final/test/test40_wobracket.json' --output_path='../../output/gemini/3_shot_sentence_40.json' --prompt_func="get_prompt_3_shot_sentence"

# 6-shot
python use_gemini.py --input_path='../../dataset/final/test/test40_wobracket.json' --output_path='../../output/gemini/6_shot_sentence_40.json' --prompt_func="get_prompt_6_shot_sentence"

# terminology-level
# seen
# vanilla
python use_gemini.py --input_path='../../dataset/final/valid.json' --output_path='../../output/gemini/vanilla_term.json' --prompt_func="get_prompt_terminology"

# 3-shot
python use_gemini.py --input_path='../../dataset/final/valid.json' --output_path='../../output/gemini/3_shot_term.json' --prompt_func="get_prompt_3_shot_terminology"

# 6-shot
python use_gemini.py --input_path='../../dataset/final/valid.json' --output_path='../../output/gemini/6_shot_term.json' --prompt_func="get_prompt_6_shot_terminology"

# terminology-level

# 3-shot
python use_gemini.py --input_path='../../dataset/final/valid.json' --output_path='../../output/gemini/3_shot_dist_term.json' --prompt_func="get_prompt_3_shot_dist_terminology" --use_similar_example 

# 3-shot (2 part)
python use_gemini.py --input_path='../../dataset/final/valid.json' --output_path='../../output/gemini/3_shot_dist_term_2_part.json' --prompt_func="get_prompt_3_shot_dist_terminology_2_part" --use_similar_example

# 6-shot
python use_gemini.py --input_path='../../dataset/final/valid.json' --output_path='../../output/gemini/6_shot_dist_term.json' --prompt_func="get_prompt_6_shot_dist_terminology" --use_similar_example 

# 3-shot (2 part)
python use_gemini.py --input_path='../../dataset/final/valid.json' --output_path='../../output/gemini/6_shot_dist_term_2_part.json' --prompt_func="get_prompt_6_shot_dist_terminology_2_part" --use_similar_example

# unseen
python use_gemini.py --input_path='../../dataset/final/unseen.json' --output_path='../../output/gemini/vanilla_term_unseen.json' --prompt_func="get_prompt_terminology"

python use_gemini.py --input_path='../../dataset/final/unseen.json' --output_path='../../output/gemini/3_shot_term_unseen.json' --prompt_func="get_prompt_3_shot_terminology"

python use_gemini.py --input_path='../../dataset/final/unseen.json' --output_path='../../output/gemini/6_shot_term_unseen.json' --prompt_func="get_prompt_6_shot_terminology"

python use_gemini.py --input_path='../../dataset/final/unseen.json' --output_path='../../output/gemini/3_shot_dist_term_unseen.json' --prompt_func="get_prompt_3_shot_dist_terminology" --use_similar_example 

python use_gemini.py --input_path='../../dataset/final/unseen.json' --output_path='../../output/gemini/6_shot_dist_term_unseen.json' --prompt_func="get_prompt_6_shot_dist_terminology" --use_similar_example 

# sentence level extract terminology 
# vanilla
python use_gemini.py --input_path='../../dataset/final/test/test40_bracket.json' --output_path='../../output/gemini/vanilla_sentence_40_bracket.json' --prompt_func="get_prompt_sentence"

# 3-shot
python use_gemini.py --input_path='../../dataset/final/test/test40_bracket.json' --output_path='../../output/gemini/3_shot_sentence_40_bracket.json' --prompt_func="get_prompt_3_shot_sentence_term"

# 6-shot
python use_gemini.py --input_path='../../dataset/final/test/test40_bracket.json' --output_path='../../output/gemini/6_shot_sentence_40_bracket.json' --prompt_func="get_prompt_6_shot_sentence_term"

```

## Eval Result

### Google Gemini
#### Sentence-level (data amount: 40) (1-step) (example: chat-gpt translation)
| prompt | BLEU | CHRF++ | COMET-22 | COMET-KIWI |
| --- | --- | --- | --- | --- |
| vanilla | 58.104 | 71.091 | 93.4 | 76.7 |
| 3-shot | 43.57 | 60.84 | 90.33 | 74.56 |
| 6-shot | 51.88 | 67.01 | 91.57 | 75.6 |
| 3-shot(\*) | 64.25 | 76.57 | 94.11 | 79.17 |
| 5-shot(\*) | 63.98 | 78.4 | 94.99 | 79.98 |
| 6-shot(\*) | 60.39 | 74.352 | 93.63 | 78.06 |
| 3-shot(\#) | 64.83 | 77.767 | 94.29 | 79.94 |
| 6-shot(\#) | 64.54 | 77.007 | 94.24 | 79.35 |

-> 用chat-gpt翻譯的結果當作example，效果更差了 (也可能是翻譯的domain不同)
-> 單獨翻譯時6-shot的進步(比起3-shot, 連江縣)，在這裡也可以看到提升

\* 用chatgpt生成相似的template，再用chatgpt翻譯
-> 就算提供的example 翻譯結果不一定對，但提供相似的句子，效果還是會比較好
-> 可能是因為提供的例子翻譯結果也不對，所以無法透過更多的例子來提升效能

\# 測試用相似的句子，但翻譯的結果不是ground truth，是用google translate翻譯的結果
-> example的部分，用google translate翻譯的結果，比起chatgpt翻譯的結果，效果更好一些
-> google translate 翻譯的結果，提供更多的例子，效果變差的比較慢

#### Sentence-level (data amount: 40) (1-step) (example: chat-gpt translation) (with bracket)
| prompt | BLEU | CHRF++ | COMET-22 | COMET-KIWI |
| --- | --- | --- | --- | --- |
| vanilla | 58.104 | 71.091 | 93.4 | 76.7 |
| 3-shot | 48.30 | 66.06 | 91.22 | 75.76 |
| 6-shot | 50.43 | 69.06 | 92.651 | 77.38 |

-> 有bracket的情況下，效果變好了一些

#### Terminology-level (data amount: 250) (seen)
| prompt | Exact-match | CHRF++ | 
| --- | --- | --- | 
| vanilla | 14.4 | 43.345 | 
| 3-shot | 52.4 | 71.951 | 
| 6-shot | 55.6 | 74.331 | 
| 3-shot w/ edit| 58.8 | 79.16 |
| 6-shot w/ edit| 60 | 79.965 |
| 3-shot w/ edit (2 part)| 52.8 | 76.971 |
| 6-shot w/ edit (2 part)| 64 | 81.937 |

(2 part) -> Higher diversity, 提升shot帶來的效果更顯著

#### Terminology-level (data amount: 44) (unseen)
| prompt | Exact-match | CHRF++ | 
| --- | --- | --- | 
| vanilla | 40.9 | 64.826 | 
| 3-shot | 54.5 | 52.477 | 
| 6-shot | 61.3 | 66.741 | 
| 3-shot w/ edit| 90.9 | 79.792 |
| 6-shot w/ edit| 93.1 | 78.861 |
| 3-shot w/ edit (2 part)| 81.81 | 76.403 |
| 6-shot w/ edit (2 part)| 95.45 | 80.928 |

#### Terminology-level (data amount: 44) (unseen) (翻譯整句，再從裡面挑出地名的部分，看看效果如何)
| prompt | Exact-match | CHRF++ | 
| --- | --- | --- | 
| vanilla | 7.7 | 29.404 | 
| 3-shot | 50 | 60.249 | 
| 6-shot | 42.5 | 64.437 | 

-> 翻譯時常常出現地名沒有用bracket包起來的情況，所以eval 結果變差了
-> 也有沒有ouput bracket包完整的情況

### Helsinki-nlp
#### Terminology-level (data amount: 250)
| method | Exact-match | CHRF++ | 
| --- | --- | --- | 
| w/o finetuning | 0.2 | 11.822 | 
| with finetuning | 55.93 | 76.244 | 

### mBart
#### Terminology-level (data amount: 250)
| method | Exact-match | CHRF++ | 
| --- | --- | --- | 
| w/o finetuning | 0.0 | 5.817 | 
| with finetuning | 63.80 | 82.998 | 

