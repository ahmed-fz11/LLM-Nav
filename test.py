import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import os
import glob
import pandas as pd

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, Goal, r2r_dataset, img_folder_path):
    setup(rank, world_size)
    
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    for index, row in r2r_dataset.iterrows():
        paths = row['path']
        generated_plan = []
        summary = "No summary yet"
        
        for path in paths:
            prompt = f"""
            "Goal":{Goal},
            "recent_summary":{summary}
            """
            start_word = path
            pattern = os.path.join(img_folder_path, f'{start_word}*')
            image_files = glob.glob(pattern)
            image_names = [os.path.basename(image) for image in image_files]
            
            chat_template = "You are an LLM Planner, designed to take in an end goal, and summary of what has happened before. This is given to you in JSON format. You are also given multiple images, which tell you about the current scene. Use these images, the end goal and the summary to give a sentence on what we should do next to achieve the end goal.###Human: " + " ".join(["<image>" for image_name in image_names]) + "\n<prompt>###Assistant:"
    
            images = [Image.open(os.path.join(img_folder_path, image_name)) for image_name in image_names]
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                },
            ]
            
            prompt = processor.apply_chat_template(conversation, chat_template=chat_template, add_generation_prompt=True)
            inputs = processor(prompt, images=images, return_tensors="pt").to(rank)
            output = model.module.generate(**inputs, max_new_tokens=100)
            hlp = processor.decode(output[0], skip_special_tokens=True)

            if hlp == "end":
                break
            
            generated_plan.append(hlp)
            # summary = summarize_experiences(summary, hlp) # Include your summary function here if needed
            
            break
        
        print("final plan = ", generated_plan)
        print("final Summary = ", summary)
        break

    cleanup()

def main():
    world_size = 2
    Goal = "I want to go to the hallway next to the kitchen."
    r2r_dataset = pd.read_json("Dataset/R2R_train.json")
    r2r_dataset = r2r_dataset[r2r_dataset['scan'] == "17DRP5sb8fy"]
    img_folder_path = 'Dataset/17DRP5sb8fy/matterport_color_images'

    mp.spawn(main_worker,
             args=(world_size, Goal, r2r_dataset, img_folder_path),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
