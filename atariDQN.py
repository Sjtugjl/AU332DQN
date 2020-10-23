# -*- coding:utf-8 -*-
# DQN homework.
import os
import sys
import gym
import time
import math
from tqdm import tqdm
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gym import wrappers
from utils import *
from queue import PriorityQueue

# hyper-parameter.
EPISODES = 500
CYF = 1  # delete when handing in.
use_gpu = True
act_rpt = 5

class DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see MsPacman learning, then change to True
        self.render = True

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.9
        self.learning_rate = 0.1
        self.step_done = 0
        self.epsilon = 0.9
        self.EPS_START = self.epsilon
        self.EPS_END = 0.1
        self.EPS_DECAY = 500
        self.batch_size = 128
        self.train_start = 1000
        self.UPDATE_TARGET_EVERY = 5

        # create replay memory using deque
        self.maxlen = 8000
        self.memory = deque(maxlen=self.maxlen)
        self.preprocess_stack = deque([], maxlen=2)
        # self.priority_memory = PriorityQueue(maxsize=self.maxlen)
        self.memory_list = []

        # create main model
        self.model_target = self.build_model()
        self.model_eval = self.build_model()

    # approximate Q function using Neural Network
    # you can modify the network to get higher reward.
    def build_model(self):
        model = Sequential()

        model.add(Dense(128, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:  # choose action randomly
            # Decay Epsilon
            agent.epsilon_decay()
            return random.randrange(self.action_size)
        else:
            # Decay Epsilon
            agent.epsilon_decay()
            q_value = self.model_eval.predict(state)
            return np.argmax(q_value[0])

    def choose_reward(self, i):
        return i[2]

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        # self.memory.append((state, action, reward, next_state, done))
        # self.priority_memory.put((-1*reward, action, state, next_state, done))
        if len(self.memory_list) < self.maxlen:
            self.memory_list.append((state, action, reward, next_state, done))
            self.memory_list.sort(key=self.choose_reward, reverse=True)
        else:
            self.memory_list[self.maxlen - 1] = (state, action, reward, next_state, done)
            self.memory_list.sort(key=self.choose_reward, reverse=True)

    def epsilon_decay(self):
        if len(self.memory) < self.train_start:
            return
        # epsilon decay.
        if self.epsilon > self.EPS_END:
            self.epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                           math.exp(-1. * self.step_done / self.EPS_DECAY)
        self.step_done += 1

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = self.memory[0:batch_size]

        # mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model_eval.predict(update_input)
        target_val = self.model_target.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        self.model_eval.fit(update_input, target, batch_size=self.batch_size,
                            epochs=1, verbose=0)

    def eval2target(self):
        self.model_target.set_weights(self.model_eval.get_weights())


if __name__ == "__main__":
    # load the gym env
    env = gym.make('MsPacman-ram-v0')
    # set  random seeds to get reproduceable result(recommended)
    set_random_seed(0)
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # create the agent
    agent = DQNAgent(state_size, action_size)
    # log the training result
    scores, episodes = [], []
    graph_episodes = []
    graph_score = []
    avg_length = 10
    sum_score = 0
    total_score = 0
    start = time.time()
    end = time.time()
    # train DQN
    # for e in tqdm(range(0, EPISODES), ascii=True, unit='episodes', file=sys.stdout):
    for e in tqdm(range(0, EPISODES), ascii=True, unit='episodes', file=sys.stdout):
        start = time.time()
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        lives = 3
        while not done:
            dead = False
            while not dead:
                reward = 0
                # render the gym env
                if agent.render:
                    env.render()
                # get action for the current state
                action = agent.get_action(state)
                # take the action in the gym env, obtain the next state
                # skip frames by repeating action
                for i in range(act_rpt):
                    next_state, _reward, done, info = env.step(action)
                    reward += _reward
                    next_state = np.reshape(next_state, [1, state_size])
                    agent.preprocess_stack.append(next_state)
                # judge if the agent dead
                dead = info['ale.lives'] < lives
                lives = info['ale.lives']
                # update score value
                score += reward

                # save the sample <s, a, r, s'> to the replay memory
                agent.append_sample(state, action, reward, next_state, done)

                # train the evaluation network
                agent.train_model()

                # go to the next state
                state = next_state

            # update the target network after some iterations.
            if e % agent.UPDATE_TARGET_EVERY == 0:
                agent.eval2target()

        # print info and draw the figure.
        if done:
            end = time.time()
            scores.append(score)
            sum_score += score
            total_score += score
            episodes.append(e)

            # plot the reward each episode
            # pylab.plot(episodes, scores, 'b')

            print("\nscore:", score, "  memory length:",
                  len(agent.memory), "  epsilon:", agent.epsilon, "average score :", float(total_score/(e+1)))
        if e % avg_length == 0:
            graph_episodes.append(e)
            graph_score.append(sum_score / avg_length)
            print(sum_score / avg_length)
            sum_score = 0

            # plot the reward each avg_length episodes
            pylab.plot(graph_episodes, graph_score, 'r')
            pylab.savefig("./pacman_avg.png")

        # save the network if you want to test it.
    # agent.model_target.save('model_{0}.h5'.format(e))