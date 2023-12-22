## Google Gemini
### set up environment
```shell
pip install -q -U google-generativeai
```

### usuage
```shell
# sentence-level

# vanilla
python use_gemini.py --input_path='valid_data.json' --output_path='gemini_vanilla.json'

# 3-shot
python use_gemini.py --input_path='valid_data.json' --output_path='gemini_3_shot.json' --prompt_func="get_prompt_3_shot"

# 6-shot
python use_gemini.py --input_path='valid_data.json' --output_path='gemini_6_shot.json' --prompt_func="get_prompt_6_shot"

# terminology-level

# vanilla
python use_gemini.py --input_path='../../dataset/final/valid.json' --output_path='../../output/gemini/vanilla_term.json' --prompt_func="get_prompt_terminology"

# 3-shot
python use_gemini.py --input_path='../../dataset/final/valid.json' --output_path='../../output/gemini/3_shot_term.json' --prompt_func="get_prompt_3_shot_terminology"

# 6-shot
python use_gemini.py --input_path='../../dataset/final/valid.json' --output_path='../../output/gemini/6_shot_term.json' --prompt_func="get_prompt_6_shot_terminology"
```

### Eval Result
| Model | BLEU | CHRF++ | COMET-22 | COMET-KIWI |
| --- | --- | --- | --- | --- |
| vanilla | 0.0 | 0.0 | 0.0 | 0.0 |
| 3-shot | 0.0 | 0.0 | 0.0 | 0.0 |
| 6-shot | 0.256 | 43.372 | 0.812 | 0.658 |

