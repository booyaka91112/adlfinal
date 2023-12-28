import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Llama 2 model and tokenizer from Hugging Face Model Hub
model_name = "booyaka91112/llama-zh-vn-PLCVD"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to generate translation
def translate_text(input_text):
    ori_input_text=input_text
    input_text="將中文翻譯成越南語。\nUSER:廣義社 ASSISTANT:Xã Quảng Nghĩa\nUSER:海防市 ASSISTANT:Thành phố Hải Phòng\nUSER:南江縣 ASSISTANT:Huyện Nam Giang\nUSER:義林坊 ASSISTANT:Xã Nghĩa Lâm\nUSER:萊州省 ASSISTANT:Tỉnh Lai Châu\nUSER:德才市鎮 ASSISTANT:Thị trấn Đức Tài\nUSER:" + input_text +" ASSISTANT:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    tmp=tokenizer(ori_input_text,return_tensors='pt',truncation=True).input_ids
    input_ids_list=input_ids.tolist()
    a=5*len(tmp[0])
    output_tokens =  model.generate(input_ids,max_new_tokens=a,repetition_penalty=1)    
    #output_ids = model.generate(input_ids, max_length=250, no_repeat_ngram_size=3)
    output_tokens_list = output_tokens[0].tolist()
    input_length = len(input_ids_list[0])
    
    generated_without_input = output_tokens_list[input_length:]
    output = tokenizer.decode(
          generated_without_input,
          skip_special_tokens=True
        )
    output = output.split('\n')[0]    
    
    #translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

# Create a Gradio interface
iface = gr.Interface(
    fn=translate_text,
    inputs="text",
    outputs="text",
    title="Llama 2 Text Translation",
    description="Translate text from Chinese to Vietnamese using Llama 2.",
)

# Launch the Gradio interface
iface.launch(share=True)
