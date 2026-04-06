from fastapi import FastAPI
from env.environments import DebugMLEnv
from env.models import Action

app = FastAPI()

env = DebugMLEnv()

@app.post('/reset')
def reset(): 
    obs = env.reset()
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
