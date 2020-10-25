import os
import sys
import gym
import time
import math
import heapq
from tqdm import tqdm
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model
from gym import wrappers
from utils import *
from queue import PriorityQueue

name = 'model_26.h5'
model = load_model(name)
# hyper-parameter.
EPISODES = 100
render = False
if __name__ == "__main__":
    # load the gym env
    env = gym.make('MsPacman-ram-v0')
    # set  random seeds to get reproduceable result(recommended)
    set_random_seed(0)
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # log the training result
    scores, episodes = [], []
    graph_episodes = []
    graph_score = []
    avg_length = 10
    sum_score = 0
    total_score = 0
    start = time.time()
    end = time.time()

    for e in tqdm(range(0, EPISODES), ascii=True, unit='episodes', file=sys.stdout):
        start = time.time()
        done = False
        score = 0
        state = env.reset()
        # print(state)
        state = np.reshape(state, [1, state_size])
        lives = 3
        frame_stack = deque(maxlen=4)
        frame_stack.append(state)

        for skip in range(90):
            env.step(0)

        while not done:
            dead = False
            while not dead:
                reward = 0
                # render the gym env
                if render:
                    env.render()
                # get action for the current state
                state = sum(frame_stack) / len(frame_stack)

                action = np.argmax(model.predict(state)[0])

                # take the action in the gym env, obtain the next state

                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                frame_stack.append(next_state)
                next_state = sum(frame_stack) / len(frame_stack)

                # judge if the agent dead
                dead = info['ale.lives'] < lives
                lives = info['ale.lives']

                # update score value
                score += reward

                # go to the next state
                state = next_state

        # print info and draw the figure.
        if done:
            end = time.time()
            scores.append(score)
            sum_score += score
            total_score += score
            episodes.append(e)

        print("\nscore:", score, "average score :", float(total_score / (e + 1)))

        if e % avg_length == 0:
            graph_episodes.append(e)
            graph_score.append(sum_score / avg_length)
            print(sum_score / avg_length)
            sum_score = 0

            # plot the reward each avg_length episodes
            pylab.plot(graph_episodes, graph_score, 'r')
            pylab.savefig("./test_{0}.png".format(name))
