## Google Gemini
### set up environment
```shell
pip install -q -U google-generativeai
```

### usuage
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
```

### Eval Result

### Google Gemini
#### Sentence-level (data amount: 40) (1-step) (example: chat-gpt translation)
| prompt | BLEU | CHRF++ | COMET-22 | COMET-KIWI |
| --- | --- | --- | --- | --- |
| vanilla | 58.104 | 71.091 | 93.4 | 76.7 |
| 3-shot | 43.57 | 60.84 | 90.33 | 74.56 |
| 6-shot | 51.88 | 67.01 | 91.57 | 75.6 |
-> 用chat-gpt翻譯的結果當作example，效果更差了 (也可能是翻譯的domain不同)
-> 單獨翻譯時6-shot的進步(比起3-shot, 連江縣)，在這裡也可以看到提升

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

