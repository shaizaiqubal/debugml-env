import os
import random
from typing import List, Optional

from openai import OpenAI

from env.environments import DebugMLEnv
from env.models import Action

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MAX_STEPS = 15
SUCCESS_THRESHOLD = 0.8


def get_api_key() -> str:
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set HF_TOKEN for the Hugging Face router or OPENAI_API_KEY before running this script."
        )
    return api_key


client = OpenAI(base_url=API_BASE_URL, api_key=get_api_key())


# ---------------------------------------------------------------------------
# Loggers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def clamp_score(score: float) -> float:
    return max(0.01, min(0.99, score))


def log_end(success: bool, steps: int, rewards: List[float], score: float):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str} score={clamp_score(score):.2f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def build_prompt(obs, last_action: Optional[str], last_reward: Optional[float]) -> str:
    current_score = round(0.5 * obs.accuracy + 0.25 * obs.precision + 0.25 * obs.recall, 2)
    target_score = 0.85

    prev_info = ""
    if last_action and last_reward is not None:
        prev_info = f""" 
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


def clean_action(output: str) -> str:
    text = output.strip().lower().replace(" ", "_")
    valid = ["add_scaling", "fix_split", "add_feature", "remove_feature"]

    for action in valid:
        if action in text:
            return action

    return random.choice(valid)


def get_action(obs, last_action: Optional[str], last_reward: Optional[float]) -> str:
    prompt = build_prompt(obs, last_action, last_reward)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2,
        max_tokens=10,
    )
    raw = response.choices[0].message.content or ""
    return clean_action(raw)


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_name: str) -> float:
    env = DebugMLEnv()
    rewards: List[float] = []
    info = {}
    last_action = None
    last_reward = None

    log_start(task=task_name, env="debugml", model=MODEL_NAME)

    try:
        obs = env.reset(task_name)

        for step_num in range(1, MAX_STEPS + 1):
            action = get_action(obs, last_action, last_reward)

            error = None
            try:
                obs, reward, done, info = env.step(Action(type=action))
            except Exception as e:
                error = str(e)
                reward = 0.0
                done = True

            rewards.append(reward)
            last_action = action
            last_reward = reward

            log_step(step=step_num, action=action, reward=reward, done=done, error=error)

            if done:
                break

    except Exception as e:
        print(f"[CRASH] {e}", flush=True)
        log_end(success=False, steps=len(rewards), rewards=rewards, score=0.01)
        return 0.01

    score = clamp_score(info["task_score"] if "task_score" in info else 0.01)
    success = score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=len(rewards), rewards=rewards, score=score)
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tasks = ["fix_basics", "optimize_features", "full_pipeline_optimization", "stability_optimization"]
    task_name = random.choice(tasks)
    run_task(task_name)


if __name__ == "__main__":
    main()
