# -*- coding: utf-8 -*-
import gym
import cv2
import wrappers
import tf_dqn
import random

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import argparse
import time
import numpy as np
from numpy import matlib
import collections
import sys


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            q_vals_v = net.model(state_a)
            action = int(tf.math.argmax(q_vals_v,1)[0])

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        exp = Experience(self.state, action, reward, is_done, new_state)            
        self.exp_buffer.append(exp)
        self.state = new_state 
        if is_done:
            done_reward = self.total_reward
            self._reset()       
        return done_reward

def get_loss(batch, tgt_net, net):
    states, actions, rewards, dones, next_states = batch
    not_dones = -1.0*np.asarray(dones) + 1.0
    rewards_v = tf.Variable(np.asarray(rewards), dtype=tf.float32)  
    not_dones_v = tf.Variable(np.asarray(not_dones), dtype=tf.float32)
    gammas_v = tf.Variable( GAMMA*np.ones(not_dones.shape), dtype=tf.float32 )
    gammas_v = tf.math.multiply(not_dones_v, gammas_v)
    next_state_input = np.asarray(next_states)
    next_state_values = tgt_net.model(next_state_input)
    next_state_values = tf.math.reduce_max(next_state_values, axis=1)
    next_state_values = tf.math.multiply(next_state_values, gammas_v)
    state_action_expected_values = tf.math.add(rewards_v, next_state_values)
    state_action_expected_values = tf.stop_gradient(state_action_expected_values)
    action_indices = np.transpose(np.concatenate((np.arange(BATCH_SIZE), actions)).reshape(2,BATCH_SIZE))
    batch_states = np.asarray(states)
    state_action_values = net.model(batch_states)
    state_action_taken_values = tf.gather_nd(state_action_values, action_indices)
    loss = tf.keras.losses.mean_squared_error(state_action_expected_values, state_action_taken_values)
    return loss
        

if __name__ == "__main__":

    print(str(tf.config.experimental.list_physical_devices('GPU')))
    
    env = wrappers.make_env('PongNoFrameskip-v4')
    net = tf_dqn.DQN((4,84,84), env.action_space.n)
    tgt_net = tf_dqn.DQN((4,84,84), env.action_space.n)

    #writer = SummaryWriter(comment="-" + args.env)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    #cv2.namedWindow("State", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("State",600,600)
    #cv2.imshow("State", agent.state[0])
    #cv2.waitKey()

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon)
        #cv2.imshow("State", agent.state[0])
        #cv2.waitKey(int(1000/90))
        
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed
            ))
            if best_mean_reward is None or best_mean_reward < mean_reward:
                #ref_net.model.set_weights(net.model.get_weights())
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > 15:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.model.set_weights(net.model.get_weights())

        batch = buffer.sample(BATCH_SIZE)

        with tf.GradientTape() as tape:
            pretrain_loss = get_loss(batch, tgt_net, net)
            
        gradients = tape.gradient(pretrain_loss, net.model.trainable_variables)
        net.optimizer.apply_gradients(zip(gradients, net.model.trainable_variables))
        
        #posttrain_loss = get_loss(batch, tgt_net, net)
        #print(str(pretrain_loss) + " " + str(posttrain_loss))
        

    cv2.destroyAllWindows()
