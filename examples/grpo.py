# train_grpo.py
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")
# dataset = Dataset.from_list(
#     [
#         {"prompt": [{"content": "What is 2+2?", "role": "user"}], "task": "math"},
#         {"prompt": [{"content": "Write a function that returns the sum of two numbers.", "role": "user"}], "task": "code"},
#         {"prompt": [{"content": "What is 3*4?", "role": "user", "role": "user"}], "task": "math"},
#         {"prompt": [{"content": "Write a function that returns the product of two numbers.", "role": "user"}], "task": "code"},
#     ]
# )


# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(prompts, completions, **kwargs):
    for prompt, completion in zip(prompts, completions):
        print(f"### Input:\n{prompt}")
        print(f"### Output:\n{completion}")
    return [-abs(20 - len(completion[0]["content"])) for completion in completions]


training_args = GRPOConfig(output_dir="outputs/Qwen2.5-0.5B-GRPO", logging_steps=1)
trainer = GRPOTrainer(model="/data/jjwang/pretrained/Qwen/Qwen2.5-0.5B-Instruct/", reward_funcs=reward_len, args=training_args, train_dataset=dataset)
trainer.train()
