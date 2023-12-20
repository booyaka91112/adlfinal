import google.generativeai as genai
import os
import argparse
import json
import utils
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./data.json', help="input file path")
    parser.add_argument('--output_path', type=str, default='x')
    parser.add_argument('--prompt_func', type=str, default='get_prompt', help="get prompt function name")
    args = parser.parse_args()
    return args

def main():
    """Use Google Gemini to translate from Chinese to Vietnamese."""
    args = parse_args()

    # set up model
    GOOGLE_API_KEY=os.getenv('PALM2_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')

    # read data
    with open(args.input_path) as f:
        data = json.load(f)

    source_list = []
    prediction_list = []
    reference_list = []
    get_prompt = getattr(utils, args.prompt_func)
    for key, value in tqdm(data.items(), total=len(data)):
        response = model.generate_content(get_prompt(key),
            generation_config=genai.types.GenerationConfig(
            # Only one candidate for now.
            candidate_count=1,
            stop_sequences=['x'],
            temperature=1.0))
        
        source_list.append(key)
        prediction_list.append(response.text)
        reference_list.append(value)

    df = pd.DataFrame({'src': source_list,
                       'mt': prediction_list,
                    'ref': reference_list})
    
    df.to_json(args.output_path, orient='records', force_ascii=False)

if __name__ == '__main__':
    main()
