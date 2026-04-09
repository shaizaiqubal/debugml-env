from fastapi import FastAPI
from env.environments import DebugMLEnv
from env.models import Action, ResetRequest
from typing import Optional

app = FastAPI()

env = DebugMLEnv()

@app.post('/reset')
def reset(request: Optional[ResetRequest] = None):
    task_name = request.task_name if request else None
    obs = env.reset(task_name)
    return obs

@app.post('/step')
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        'Observation' : obs,
        'Reward' : reward,
        'Done' : done,
        'Info' : info
    }

@app.get('/state')
def state():
    return env.state()

@app.get("/")
def root():
    return {"message": "DebugML API is running"}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
