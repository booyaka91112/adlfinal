import pandas as pd

# Read the Excel file
data = pd.read_excel('dataset_combined.xlsx')

# Convert the DataFrame to JSON format
json_data = data.to_dict(orient='records')

# Save as JSON file
import json
with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)
