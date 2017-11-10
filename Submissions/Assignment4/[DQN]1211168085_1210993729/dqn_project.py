from __future__ import print_function
import gym
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from itertools import count
from replay_memory import ReplayMemory, Transition
import env_wrappers
import random
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action="store_true", default=False, help='Run in eval mode')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)


class DQN(object):
    """
    A starter class to implement the Deep Q Network algorithm
    TODOs specify the main areas where logic needs to be added.
    If you get an error a Box2D error using the pip version try installing from source:
    > git clone https://github.com/pybox2d/pybox2d
    > pip install -e .
    """

    def __init__(self, env):

        self.env = env
        self.sess = tf.Session()

        # A few starter hyperparameters
        self.batch_size = 128
        self.gamma = 0.99
        # If using e-greedy exploration
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000  # in episodes
        # If using a target network
        self.clone_steps = 5000

        # memory
        self.replay_memory = ReplayMemory(100000)
        # Perhaps you want to have some samples in the memory before starting to train?
        self.min_replay_size = 10000

        # define yours training operations here...

        self.observation_input = tf.placeholder(tf.float32, shape=[None] + list(self.env.observation_space.shape))
        self.q_values = self.build_model(self.observation_input)

        # define your update operations here...

        self.num_episodes = 0
        self.num_steps = 0

        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, observation_input, scope='train'):
        """
        TODO: Define the tensorflow model
        Hint: You will need to define and input placeholder and output Q-values
        Currently returns an op that gives all zeros.
        use relu for nonlinear function
        2^n neurons in each layer- 2 layers.
        input * weight vector + bias -> pass inside relu
        observation_input:
        Tensor("Placeholder:0", shape=(?, 8), dtype=float32)
        dimensions:
        observation_input : 64 * 8
        x: 4 * ?
        weight: 8 * 256
        bias : updated during back propogation
        neurons: 256
        """
        print("in build_model")
        # weight1 = tf.Variable(tf.random_uniform([8, 256], 0, 0.01, dtype=tf.float32))
        # weight2 = tf.Variable(tf.random_uniform([256, 256], 0, 0.01, dtype=tf.float32))
        # weight3 = tf.Variable(tf.random_uniform([256, 4], 0, 0.01, dtype=tf.float32))
        hidden1 = layers.fully_connected(observation_input, 128, activation_fn=tf.nn.relu,
                                         biases_initializer=tf.zeros_initializer())
        hidden2 = layers.fully_connected(hidden1, 128, activation_fn=tf.nn.relu,
                                         biases_initializer=tf.zeros_initializer())
        q_values = layers.fully_connected(hidden2, env.action_space.n, activation_fn=None)
        # hidden1 = tf.matmul(observation_input, weight1) # + bias1
        # hidden2 = tf.matmul(hidden1, weight2) # + bias2
        # q_values = tf.matmul(hidden2, weight3) # + bias3
        print(q_values)
        time.sleep(5)

        with tf.variable_scope(scope):
            return q_values

    def select_action(self, obs, evaluation_mode=False):
        """
        TODO: Select an action given an observation using your model. This
        should include any exploration strategy you wish to implement
        If evaluation_mode=True, then this function should behave as if training is
        finished. This may be reducing exploration, etc.
        Currently returns a random action.
        """
        print("in select_action function")
        probability = 0.8
        random_choice = random.random()
        if random_choice < probability:
            print("in exploitation")
            # max_q = tf.argmax(self.q_values, axis=1)
            # print(max_q)
            print("self.q_values")
            print(self.q_values)
            return np.argmax(self.sess.run(self.q_values, feed_dict={self.observation_input: obs})[0])
        else:
            print("in exploration")
            return env.action_space.sample()

    def update(self, obs):
        """
        TODO: Implement the functionality to update the network according to the
        Q-learning rule
        tensorflow gradient calculation
        update weight and bias
        iterate over replay_memory
        """
        # q_values = self.build_model(obs)
        # update q values
        # hidden1 = layers.fully_connected(obs, 128, activation_fn=tf.nn.relu,
        #                                  biases_initializer=tf.zeros_initializer())
        # hidden2 = layers.fully_connected(hidden1, 128, activation_fn=tf.nn.relu,
        #                                  biases_initializer=tf.zeros_initializer())
        # self.q_values = layers.fully_connected(hidden2, env.action_space.n, activation_fn=None)
        # raise NotImplementedError
        self.sess.run(self.q_values, feed_dict={self.observation_input: obs})[0]
        self.sess.run(self.update_op, feed_dict={self.observation_input: observations, self.q_target: q_target})


def train(self):
    """
    The training loop. This runs a single episode.
    TODO: Implement the following as desired:
        1. Storing transitions to the ReplayMemory
        2. Updating the network at some frequency
        3. Backing up the current parameters to a reference, target network
    """
    done = False
    obs = env.reset()
    obs = obs.reshape([1, 8])
    print("in train function")
    while not done:
        action = self.select_action(obs, evaluation_mode=False)
        # orig: next_obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        obs = obs.reshape([1, 8])
        self.update(obs)
        self.num_steps += 1
    self.num_episodes += 1


def eval(self, save_snapshot=True):
    """
    Run an evaluation episode, this will call
    """
    total_reward = 0.0
    ep_steps = 0
    done = False
    obs = env.reset()
    print("in eval function")
    while not done:
        obs = obs.reshape([1, 8])
        env.render()
        action = self.select_action(obs, evaluation_mode=True)
        obs, reward, done, info = env.step(action)
        # push all this in replay memory
        total_reward += reward
    print("Evaluation episode: ", total_reward)
    if save_snapshot:
        print("Saving state with Saver")
        self.saver.save(self.sess, 'models/dqn-model', global_step=self.num_episodes)


def train(dqn):
    for i in count(1):
        dqn.train()
        # every 10 episodes run an evaluation episode
        if i % 10 == 0:
            dqn.eval()


def eval(dqn):
    """
    Load the latest model and run a test episode
    """
    ckpt_file = os.path.join(os.path.dirname(__file__), 'models/checkpoint')
    with open(ckpt_file, 'r') as f:
        first_line = f.readline()
        model_name = first_line.split()[-1].strip("\"")
    dqn.saver.restore(dqn.sess, os.path.join(os.path.dirname(__file__), 'models/' + model_name))
    dqn.eval(save_snapshot=False)


if __name__ == '__main__':
    # On the LunarLander-v2 env a near-optimal score is some where around 250.
    # Your agent should be able to get to a score >0 fairly quickly at which point
    # it may simply be hitting the ground too hard or a bit jerky. Getting to ~250
    # may require some fine tuning.
    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    # Consider using this for the challenge portion
    # env = env_wrappers.wrap_env(env)

    dqn = DQN(env)
    if args.eval:
        eval(dqn)
    else:
        train(dqn)