'''
Apply q_learning to selve MountainCar-v0
env info:
action space [0,1]  0 for left 1 for right
observation space [x,v] x for position v for velocity

to do what:
learn QTable   f(s,a)
| s  | a1 | a2
| s0 |    |

obstacle:
obs space is not discrete
'''

import gym
import numpy as np
import time
import logging
import math

env_mcar = 'MountainCar-v0'
env = gym.make(env_mcar)
print("what we see of low", env.observation_space.low)
print("what we see of high", env.observation_space.high)

logging.basicConfig(filename="./data/q_learning_1.log", level=logging.INFO)

class RL_QLearning(object):
    def __init__(self, envname, learning_rate=0.1, gamma=0.99, epsilon=0.8):
        self.env = gym.make(envname)
        self.q_table = dict()
        self.lr = learning_rate
        self.gamma = gamma
        self.ep = epsilon

    def set_ep(self,ep):
        self.ep = ep

    def reset_env(self):
        return self.env.reset()

    def render_env(self):
        self.env.render()

    def step(self, action):
        return self.env.step(action)

    def is_new(self,s):
        return self.discrete_obs(s) in self.q_table

    def discrete_obs(self, s):
        x_l, v_l = self.env.observation_space.low
        x_h, v_h = self.env.observation_space.high
        x_ret = int(20*(s[0]-x_l)/(x_h-x_l))  # discrete to [0,40]
        v_ret = int(20*(s[-1]-v_l)/(x_h-x_l)) #
        return x_ret, v_ret

    def learn(self, s, a, r, s_, isterminal=True):
        x, v = self.discrete_obs(s)
        if (x, v) in self.q_table:
            q_predict = max(self.q_table.get((x, v)))
        else:
            self.q_table[(x, v)] = [0, 0, 0]
            q_predict = 0
        if not isterminal:
            x_, v_ = self.discrete_obs(s_)
            if (x_, v_) in self.q_table:
                q_target = r + self.gamma * max(self.q_table[(x_, v_)])
            else:
                q_target = r
        else:
            q_target = r
        self.q_table[(x, v)][a] += self.lr*(q_target - q_predict)

    def choose_action(self, s):
        # print("s:", s, self.discrete_obs(s))
        if self.discrete_obs(s) in self.q_table and np.random.uniform() < self.ep:
            qa_lst = self.q_table[self.discrete_obs(s)]
            a_chosed = qa_lst.index(max(qa_lst))
        else:
            a_chosed = np.random.choice([0, 1, 2])
        return a_chosed

if __name__ =="__main__":

    n_episode = 5000
    agent = RL_QLearning(envname=env_mcar)
    mean_10 = []

    for i_episode in range(n_episode):
        max_x = 0
        obs = agent.reset_env()
        new_ep = 0.8 + 0.2**((n_episode-i_episode)//500)
        agent.set_ep(new_ep)
        acc_r = 0
        max_r = -float('inf')
        acc_step = 0
        action_set = dict()
        while True:
            agent.render_env()
            action = agent.choose_action(obs)
            if action not in action_set:
                action_set[action] = 1
            else:
                action_set[action] += 1
            obs_, reward, done, _ = agent.step(action)
            reward += 100*math.fabs(math.fabs(obs_[0])-math.fabs(obs[0]))\
                      /(env.observation_space.high[0]-env.observation_space.low[0])
            if reward>0.2:
                reward = 0.2
            if agent.is_new(obs_):
                reward += 0.1
            if math.fabs(obs_[0]) > max_x:
                max_x = math.fabs(obs_[0])
            if reward>max_r:
                max_r =reward
            agent.learn(obs, action, reward, obs_, isterminal=done)
            acc_r += reward
            acc_step += 1
            obs = obs_
            if done:
                if len(mean_10) < 10:
                    mean_10.append(max_x)
                else:
                    mean_10[i_episode%10]=max_x
                info = "episode {} step:{} ep:{:.4f} we get:{:.2f} max_r {:.2f} max x :{:.2f}, max-mean7: {:.2f},action: {}".format(i_episode,acc_step,new_ep,acc_r,max_r,max_x,np.mean(mean_10),action_set)
                print(info)
                logging.info(info)
                if i_episode % 100 == 0:
                    tmp ="{}".format(agent.q_table)
                    logging.critical(tmp)
                break

        time.sleep(0.1)