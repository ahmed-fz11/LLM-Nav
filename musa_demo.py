from pydantic import BaseModel
import openai
import pandas as pd
import os
import glob

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import transformers
import torch
from PIL import Image

class VLMResponse(BaseModel):
    plan: str
    next_image_frame_id: int

# LLM Code  

# Initialize OpenAI API with your API key
openai.api_key = "YOUR_API_KEY"

def get_gpt_response(input_text) -> VLMResponse:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # You can change the model to gpt-3.5-turbo or another if needed
            messages=[
                {"role": "system", "content": "You are an indoor planner LLM, given a prompt with the goal,current_instruction,visible_objects and recent summary. Generate a High Level plan and the id of the next image frame to go to. If you think you have reached your goal the plan should just be the word 'end' "},  # Optional system message
                {"role": "user", "content": input_text}
            ],
            max_tokens=100,  # Limit the response length
            response_format=VLMResponse,
        )

        # Output the GPT response
        output_text = response.choices[0].message.parsed
        return output_text
    except Exception as e:
        print(f"Error with GPT API request: {e}")
        return None
    
def summarize_experiences(past_summary,hlp):
    # input_text = {"past_summary":past_summary,"plan to add":hlp}
    try:
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5",  # You can change the model to gpt-3.5-turbo or another if needed
        #     messages=[
        #         {"role": "system", "content": "You are an indoor planner LLM, given a past summary and a plan to add to that summary you must generate a summary with the new plan added in the past experince"},  # Optional system message
        #         {"role": "user", "content": input_text}
        #     ],
        #     max_tokens=100,  # Limit the response length
        # )

        # # Output the GPT response
        # output_text = response.choices[0].message
        # # return output_text

        model_id = "meta-llama/Meta-Llama-3-8B"

        # Load the model and tokenizer
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1,
        )

        # Define your system prompt
        system_prompt = "You are a summarizer. You are given a high level navigation plan, and summary of the previous executed plan. Generate a new Summary based on these two inputs"

        # Define the user prompt
        user_prompt = f"Previous Summary:{past_summary}.\nNew high level plan:{hlp}"

        # Combine the system and user prompts
        input_text = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"

        # Generate the response
        response = pipeline(input_text, max_length=100, do_sample=True)

        # Print the output
        print("summary",response[0]['generated_text'])
        return response[0]['generated_text']
    except Exception as e:
        print(f"Error with GPT API request: {e}")
        return None


def Llava_generation(input_text,image_names,base_path):
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-next-110b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-next-110b-hf", torch_dtype=torch.float16, device_map="auto") 

    # prepare image and text prompt, using the appropriate prompt template
    
    images = [Image.open(os.path.join(base_path, image_name)) for image_name in image_names]

    # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image") 
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are an LLM Planner, design to take in an end goal,and summary of what has happened before.This is given to you in JSON format. You are also given multiple images, which tell you about the current scene. Use these images, the end goal and the summary to give a sentence on what we should do next to acheive the end goal."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": input_text},
                {"type": "image"}]
        },
    ]
   
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(prompt, images=images, return_tensors="pt").to(model.device)

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)

    print("llava response = ",processor.decode(output[0], skip_special_tokens=True))
    return processor.decode(output[0], skip_special_tokens=True)


# Only checking for first training case
Goal = "I want to go to the hallway next to the kitchen."
r2r_dataset = pd.read_json("Dataset/R2R_train.json")

r2r_dataset = r2r_dataset[r2r_dataset['scan'] == "17DRP5sb8fy"]

img_folder_path = 'Dataset/17DRP5sb8fy/matterport_color_images'


for index, row in r2r_dataset.iterrows():
    # Get relevant images for the data Point
    paths = row['path']
    
    generated_plan = []
    summary = ""
    
    
    for path in paths:
        prompt = {
        "Goal":Goal,
        "recent_summary":summary
        }
        
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
    
    # Check generated_plan with r2r_dataset.iloc[index]['instructions'] 
    # Compute metrics
        
    break











# img_dataset = []
# Goal = "Find the apple"
# instruction = "Go straight then walk toward the round table."
# visible_objects = ["apple","mango","banana"]
# Summary = ""

# goal_reached = False

# generated_plan = []

# i=0
# while i!=1:
#     i=1
#     prompt = {
#     "Goal":Goal,
#     "current_instruction":instruction,
#     "visible_objects":visible_objects.join(','),
#     "recent_summary":Summary
#     }

#     response = get_gpt_response(prompt)
#     hlp = response.plan
#     next_img = img_dataset[response.next_image_frame_id]
#     generated_plan.append(hlp)
    
#     if(hlp == "end"):
#         goal_reached = True
#     else:
#         Summary = summarize_experiences(Summary,hlp)
#         instruction = hlp
#         # visible_objects = Object_detector[next_img]
        
