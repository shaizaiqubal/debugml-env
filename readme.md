# DebugML — OpenEnv Environment

An OpenEnv-compatible reinforcement learning environment that simulates intelligent debugging and optimization of machine learning pipelines.

The agent iteratively improves a pipeline by selecting actions such as fixing data splits, applying scaling, and adjusting features, guided by reward feedback and task-specific evaluation.

## Environment Description

The agent is placed in a simulated ML pipeline with suboptimal configuration — wrong train/test split, missing feature scaling, or too many/few features. The agent must identify and fix these issues to maximize a composite performance score (accuracy, precision, recall)

This environment simulates a real-world task: debugging and optimizing an ML pipeline, which is a common problem in data science workflows.

## Observation Space

| Field | Type | Description |
|---|---|---|
| accuracy | float | Current model accuracy (0.0–1.0) |
| precision | float | Current model precision |
| recall | float | Current model recall |
| scaling | bool | Whether feature scaling is applied |
| feature_count | int | Number of features (1–6) |
| test_split | float | Train/test split ratio |
| model_type | str | Model type: linear, svm, or tree |

## Action Space

| Action | Description |
|---|---|
| add_scaling | Apply feature scaling to the pipeline |
| fix_split | Correct the train/test split to 0.2 |
| add_feature | Add a feature to the pipeline |
| remove_feature | Remove a feature from the pipeline |

## Tasks

Each task defines a different initial state and evaluation objective, testing the agent’s ability to handle diverse optimization scenarios.

| Task | Difficulty | Goal |
|---|---|---|
| fix_basics | Easy | Enable scaling and fix a bad split |
| optimize_features | Medium | Tune feature count with scaling already applied |
| full_pipeline_optimization | Hard | Fix everything from a random starting state |
| stability_optimization | Hard | Maintain accuracy with minimal unnecessary steps |


## Agent Behavior

The agent uses an LLM to:

- Evaluate multiple possible actions before selecting one
- Avoid repeating actions that previously reduced performance
- Track progress toward a target score

This enables structured decision-making rather than random exploration.

## Reward

Reward is computed as:

- **Progress reward:** change in pipeline score between steps
- **Penalty:** applied for redundant or harmful actions (e.g., repeating ineffective actions)
- **Bonus:** small reward for reaching high accuracy (>0.9)

This creates a dense reward signal that encourages efficient and meaningful improvements.

**Note:**  
The environment uses two scoring systems:
- **Raw score** (accuracy-based): used internally for reward calculation and episode termination
- **Task score** (grader output): used for final evaluation, incorporating efficiency and task-specific criteria

## Setup Instructions

```bash
git clone https://github.com/shaizaiqubal/debugml-env
cd debugml-env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```
Requires Python 3.10+

## Environment Variables

| Variable | Description |
|---|---|
| API_BASE_URL | LLM API endpoint (default: HuggingFace router) |
| MODEL_NAME | Model identifier (default: Qwen/Qwen2.5-72B-Instruct) |
| HF_TOKEN | Your Hugging Face API key |

## Run Inference

```bash
export HF_TOKEN=your_token_here
python inference.py
```

## API Endpoints

- POST /reset — Reset environment, returns initial observation
- POST /step — Take an action, returns (observation, reward, done, info)
- GET /state — Returns current environment state

## Docker

```bash
docker build -t debugml .
docker run -e HF_TOKEN=your_token -p 7860:7860 debugml
```

This environment is designed as a foundation for real-world AutoML systems, where simulated scoring can be replaced with actual model training and evaluation.