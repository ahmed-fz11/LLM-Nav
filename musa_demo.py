from openai import OpenAI
import pandas as pd
import os
import glob

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

CLIENT = OpenAI(api_key="add api key here")

def summarize_experiences(past_summary,hlp):
    input_text = f"Past Summary: {past_summary}\nNew plan: {hlp}"
    try:
        response = CLIENT.chat.completions.create(
            model='gpt-3.5-turbo',
             messages=[
                {"role": "system", "content": 
                 """You are a summarizer which gives an answer in past tense.Given a past summary and and recent actions you must combine this into a single summary of past experiences. Give your answer in past tense, telling me what I have done till now.
                Use the format:
                Summary:"""},  # Optional system message
                {"role": "user", "content": input_text}
            ],
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        # Output the GPT response
        output_text = response.choices[0].message.content
        return output_text.split('Summary:')[1].strip()
    except Exception as e:
        print(f"Error with GPT API request: {e}")
        return None


def Llava_generation(input_text,image_names,base_path):
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True)
    # model.to(0) # 4bit and 8bit quantization automatically puts model on GPU

    # prepare image and text prompt, using the appropriate prompt template
    image_names.sort()
    images = [Image.open(os.path.join(base_path, image_name)) for image_name in image_names]

    # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image")
    conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": input_text},
          {"type": "image"},
          {"type": "image"},
          {"type": "image"}
        ],
    },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    print("prompt = ",prompt)
    inputs = processor(prompt, images=[images[7],images[6],images[10]], return_tensors="pt").to(0)

    # autoregressively complete prompt
    output = model.generate(**inputs,max_new_tokens=1000)
    
    result = processor.decode(output[0], skip_special_tokens=True).split("[/INST]")[1]
#     print("llava response = ",result)
    return result.strip()


# Only checking for first training case
Goal = "I want to go to the hallway next to the kitchen."
r2r_dataset = pd.read_json("Dataset/R2R_train.json")

r2r_dataset = r2r_dataset[r2r_dataset['scan'] == "17DRP5sb8fy"]

img_folder_path = 'Dataset/17DRP5sb8fy/matterport_color_images'


for index, row in r2r_dataset.iterrows():
    # Get relevant images for the data Point
    paths = row['path']

    generated_plan = []
    summary = "This is the first time, just summarize the new high level plan"


    for path in paths:
        prompt = f"""My Goal is :{Goal},use the images provided to get a scene understanding .tell me what the next step I need to take to navigate to my goal, in one sentence"""

        start_word = path
        pattern = os.path.join(img_folder_path, f'{start_word}*')
        image_files = glob.glob(pattern)
        image_names = [os.path.basename(image) for image in image_files]

        hlp = Llava_generation(prompt,image_names,img_folder_path)

        if(hlp == "end"):
            break

        generated_plan.append(hlp)
        summary = summarize_experiences(summary,hlp)

        break

    print("final plan = ",generated_plan)
    print("final Summary = ",summary)

    # TODO:
    # Check generated_plan with r2r_dataset.iloc[index]['instructions']
    # Compute metrics

    break