import os
import random
import requests
from openai import OpenAI
from typing import List, Optional

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN     = os.environ.get("HF_TOKEN", "dummy-key")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SPACE_URL    = os.environ.get("SPACE_URL", "https://shae2977-debugml-env.hf.space")

MAX_STEPS = 15
SUCCESS_THRESHOLD = 0.8

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Loggers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def reset_env(task_name: str) -> dict:
    response = requests.post(
        f"{SPACE_URL}/reset",
        json={"task_name": task_name},
        timeout=30,
    )
    return response.json()

def step_env(action: str) -> dict:
    response = requests.post(
        f"{SPACE_URL}/step",
        json={"type": action},
        timeout=30,
    )
    return response.json()

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def build_prompt(obs: dict, last_action: Optional[str], last_reward: Optional[float]) -> str:
    accuracy = obs.get("accuracy", 0)
    precision = obs.get("precision", 0)
    recall = obs.get("recall", 0)
    scaling = obs.get("scaling", False)
    feature_count = obs.get("feature_count", 3)
    model_type = obs.get("model_type", "linear")
    current_score = round(0.5 * accuracy + 0.25 * precision + 0.25 * recall, 2)

    prev_info = ""
    if last_action and last_reward is not None:
        prev_info = f"\nPrevious action: {last_action}\nPrevious reward: {last_reward}"

    return f"""You are an agent optimizing a machine learning pipeline.

Current state:
- Accuracy: {accuracy}
- Scaling applied: {scaling}
- Feature count: {feature_count}
- Model type: {model_type}
- Current score: {current_score}
- Target score: 0.85
- Gap: {round(0.85 - current_score, 2)}
{prev_info}

Available actions:
- add_scaling (only if scaling is False)
- fix_split
- add_feature (only if feature_count < 6)
- remove_feature (only if feature_count > 1)

Constraints:
- Do NOT choose add_scaling if scaling is True
- Do NOT choose add_feature if feature_count >= 6
- Do NOT choose remove_feature if feature_count <= 1
- Do not repeat an action that returned negative reward

Respond with ONLY the action name, nothing else."""

def get_action(obs: dict, last_action: Optional[str], last_reward: Optional[float]) -> str:
    valid = ["add_scaling", "fix_split", "add_feature", "remove_feature"]
    try:
        prompt = build_prompt(obs, last_action, last_reward)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=10,
        )
        content = response.choices[0].message.content
        text = content.strip().lower().replace(" ", "_") if content else ""
        for action in valid:
            if action in text:
                return action
    except Exception:
        pass
    return random.choice(valid)

# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_name: str) -> float:
    rewards: List[float] = []
    last_action = None
    last_reward = None

    log_start(task=task_name, env="debugml", model=MODEL_NAME)

    try:
        obs = reset_env(task_name)

        for step_num in range(1, MAX_STEPS + 1):
            action = get_action(obs, last_action, last_reward)

            error = None
            try:
                result = step_env(action)
                reward = float(result.get("Reward", 0.01))
                reward = max(0.01, min(0.99, reward))
                done = result.get("Done", False)
                obs = result.get("Observation", obs)
            except Exception as e:
                reward = 0.01
                done = True
                error = str(e)

            rewards.append(reward)
            last_action = action
            last_reward = reward

            log_step(step=step_num, action=action, reward=reward, done=done, error=error)

            if done:
                break

    except Exception as e:
        print(f"[CRASH] {e}", flush=True)
        log_end(success=False, steps=len(rewards), rewards=rewards)
        return 0.01

    score = sum(rewards) / len(rewards) if rewards else 0.01
    score = max(0.01, min(0.99, score))
    success = score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=len(rewards), rewards=rewards)
    return score

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tasks = ["fix_basics", "optimize_features", "full_pipeline_optimization", "stability_optimization"]
    scores = {}

    for task in tasks:
        scores[task] = run_task(task)
        print("", flush=True)  # clean break between tasks

    print("\nSCORES SUMMARY", flush=True)
    for task, score in scores.items():
        print(f"  {task:35} → {score:.3f}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"  AVERAGE                             → {avg:.3f}", flush=True)

if __name__ == "__main__":
    main()