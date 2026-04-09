#REMOVE PRINT OBS AND INFO
from openai import OpenAI
from env.environments import DebugMLEnv
from env.models import Action
import os
import random


client = OpenAI(
    base_url = os.getenv('API_BASE_URL', 'https://router.huggingface.co/v1'),
    api_key = os.getenv('HF_TOKEN')
)

env = DebugMLEnv()
tasks = ['fix_basics', 'optimize_features', 'full_pipeline_optimization', 'stability_optimization']
task_name = random.choice(tasks)
obs = env.reset(task_name)
last_action = None
last_reward = None

print(f"[START] task={task_name} env=debugml model={os.getenv('MODEL_NAME', 'Qwen/Qwen2.5-72B-Instruct')}")

def build_prompt(obs,last_action,last_reward):

    current_score = round(0.5*obs.accuracy + 0.25*obs.precision + 0.25*obs.recall, 2)
    target_score = 0.85

    prev_info = ""
    if last_action and last_reward is not None:
        prev_info =f""" 
        Previous info:
        last action: {last_action}
        last reward: {last_reward}"""

    return f"""
        You are an agent optimizing a machine learning pipeline. Your goal is to maximize accuracy by selecting the best action at each step.

        Current state:
        - Accuracy: {obs.accuracy}
        - Scaling applied: {obs.scaling}
        - Feature count: {obs.feature_count}
        - Model type: {obs.model_type}

        {prev_info}

        Score:
        - current_score: {current_score}
        - target_score: {target_score}
        - gap: {round(target_score - current_score, 2)}


        Available actions:
        - add_scaling → enables feature scaling (only useful if scaling is False)
        - fix_split → adjusts train/test split
        - add_feature → adds a feature (only useful if feature_count < 6)
        - remove_feature → removes a feature (only useful if feature_count > 1)

        Constraints you must respect:
        - If scaling is True, do NOT choose add_scaling
        - If feature_count >= 6, do NOT choose add_feature
        - If feature_count <= 1, do NOT choose remove_feature
        - Do not repeat an action that returned a negative reward

        Reason step by step:
        1. Look at the current state
        2. Eliminate any actions that violate the constraints above
        3. From the remaining actions, pick the one most likely to improve accuracy
        4. Focus on actions that reduce the score gap.
        5. If score is already high (>0.85), avoid unnecessary changes.
        6. If the pipeline is already good, avoid unnecessary changes.
        7. Respond with ONLY the action name, nothing else.

        Action:
        """



def clean_action(output):
    text = output.strip().lower().replace(" ", "_")
    valid = ["add_scaling", "fix_split", "add_feature", "remove_feature"]
    
    for action in valid:
        if action in text:  # substring match, not exact
            return action
    
    return random.choice(valid)  # random fallback, not always add_scaling


rewards = []
max_steps=15
info = {}  

for i in range(max_steps):
    prompt = build_prompt(obs, last_action, last_reward)
    response = client.chat.completions.create(
        model=os.getenv('MODEL_NAME', 'Qwen/Qwen2.5-72B-Instruct'),
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature = 0.2,
        max_tokens=10,
        
    )
    raw = response.choices[0].message.content
    action = clean_action(raw)
    
    error = "null"
    try:
        obs, reward, done, info = env.step(Action(type=action))
        rewards.append(reward)
    except Exception as e:
        error = str(e)
        reward = 0.0
        done = True
        rewards.append(reward)
    last_action=action
    last_reward=reward
    print(f"[STEP] step={i+1} action={action} reward={reward:.2f} done={str(done).lower()} error={error}")

    if done==True:
        break

success = info["task_score"] >= 0.8 if "task_score" in info else False

rewards_str = ",".join(f"{r:.2f}" for r in rewards)

print(f"[END] success={str(success).lower()} steps={len(rewards)} rewards={rewards_str} score={info.get('task_score', 0.01)}")
