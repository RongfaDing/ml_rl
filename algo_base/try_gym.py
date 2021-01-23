import gym
import time

env_name = 'Breakout-v0'
env_walkman ='BipedalWalker-v3'
env = gym.make(env_walkman)
n_episodes = 100


for i_episode in range(n_episodes):
    acc_r = 0
    acc_step = 0
    obs = env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        acc_r += reward
        acc_step += 1
        time.sleep(0.01)
        if done:
            print("failed !!")
            break
    print("total step:{} reward:{}".format(acc_step,acc_r))

