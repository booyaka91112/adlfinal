## Google Gemini
### set up environment
```shell
pip install -q -U google-generativeai
```

### usuage
```shell
# vanilla
python use_gemini.py --input_path='valid_data.json' --output_path='gemini_vanilla.json'

# 3-shot
python use_gemini.py --input_path='valid_data.json' --output_path='gemini_3_shot.json' --prompt_func="get_prompt_3_shot"

# 6-shot
python use_gemini.py --input_path='valid_data.json' --output_path='gemini_6_shot.json' --prompt_func="get_prompt_6_shot"
```

