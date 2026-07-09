from env.env import Env
from model.actions import ACTION1

env = Env()
env.reset(env.games()[0])

# Take a few actions
for _ in range(10):
    env.step(ACTION1)

print(env.arc.get_scorecard())