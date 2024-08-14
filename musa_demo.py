from pydantic import BaseModel
import openai

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

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
    input_text = {"past_summary":past_summary,"plan to add":hlp}
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5",  # You can change the model to gpt-3.5-turbo or another if needed
            messages=[
                {"role": "system", "content": "You are an indoor planner LLM, given a past summary and a plan to add to that summary you must generate a summary with the new plan added in the past experince"},  # Optional system message
                {"role": "user", "content": input_text}
            ],
            max_tokens=100,  # Limit the response length
        )

        # Output the GPT response
        output_text = response.choices[0].message
        return output_text
    except Exception as e:
        print(f"Error with GPT API request: {e}")
        return None


def Llava_generation(input_text):
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-next-110b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-next-110b-hf", torch_dtype=torch.float16, device_map="auto") 

    # prepare image and text prompt, using the appropriate prompt template
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)

    # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image") 
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": input_text},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(prompt, image, return_tensors="pt").to(model.device)

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)

    print(processor.decode(output[0], skip_special_tokens=True))

    
img_dataset = []
Goal = "Find the apple"
instruction = "Go straight then walk toward the round table."
visible_objects = ["apple","mango","banana"]
Summary = ""

goal_reached = False

generated_plan = []

i=0
while i!=1:
    i=1
    prompt = {
    "Goal":Goal,
    "current_instruction":instruction,
    "visible_objects":visible_objects.join(','),
    "recent_summary":Summary
    }

    response = get_gpt_response(prompt)
    hlp = response.plan
    next_img = img_dataset[response.next_image_frame_id]
    generated_plan.append(hlp)
    
    if(hlp == "end"):
        goal_reached = True
    else:
        Summary = summarize_experiences(Summary,hlp)
        instruction = hlp
        # visible_objects = Object_detector[next_img]
        
