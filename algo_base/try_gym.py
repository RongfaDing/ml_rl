import gym
import time

env_name = 'Breakout-v0'
env_walkman = 'BipedalWalker-v3'
env_mcar = 'MountainCar-v0'

env = gym.make(env_mcar)
n_episodes = 100
print("action space:", env.action_space, env.action_space.shape)
print("what we see of low", env.observation_space.low)
print("what we see of high", env.observation_space.high)


def discrete_obs(s):
    x_l, v_l = env.observation_space.low
    x_h, v_h = env.observation_space.high
    x_ret = int(40 * (s[0] - x_l) / (x_h - x_l))  # discrete to [0,40]
    v_ret = int(40 * (s[-1] - v_l)/ (x_h - x_l)) #
    return x_ret, v_ret

for i_episode in range(n_episodes):
    acc_r = 0
    acc_step = 0
    obs = env.reset()
    while True:
        env.render()
        print("what we see:{} -->{}".format(obs, discrete_obs(obs)))
        action = env.action_space.sample()
        print("what we do:", action)
        obs, reward, done, _ = env.step(action)
        acc_r += reward
        acc_step += 1
        time.sleep(0.01)
        if done:
            print("failed !!")
            break
    print("total step:{} reward:{}".format(acc_step, acc_r))

