import random
import gym
import numpy as np
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import datetime

from envs.world_env import WormWorldEnv

from time import time

# For training the agent:
# https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym

class WormAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)    # what is deque?
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(12, input_dim=self.state_size, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def save_model(self,dir_name=None):
        if dir_name is None:
            dir_name = ''

        self.model.save(dir_name + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.h5')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.expand_dims(state,axis=0))
        # print(act_values)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        loss = 0
        minibatch = random.sample(self.memory, batch_size)

        # with tf.device('/cpu:0'):

        for state, action, reward, next_state, done in minibatch:
            # t = time()
            target = reward
            # print('training')
            # print(state)
            # print(np.array(next_state).shape)
            # print(self.model.predict(np.expand_dims(next_state,axis=0)))
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.expand_dims(next_state,axis=0))[0])
            # print('target time: ' + str(time() - t))

            target_f = self.model.predict(np.expand_dims(state,axis=0))
            target_f[0][action] = target
            # print('target_f time: ' + str(time() - t))
            history = self.model.fit(np.expand_dims(state,axis=0), target_f, epochs=1, verbose=0)
            # print('history time: ' + str(time() - t))

            # print(history.history['loss'])
            loss += history.history['loss'][-1]

            # print('iteration time: ' + str(time() - t))

        # t = time()
        # states = None
        # targets = None
        # for state, action, reward, next_state, done in minibatch:
        #     # t = time()
        #     target = reward
        #     # print('training')
        #     # print(state)
        #     # print(np.array(next_state).shape)
        #     # print(self.model.predict(np.expand_dims(next_state,axis=0)))
        #     if not done:
        #         target = reward + self.gamma * np.amax(self.model.predict(np.expand_dims(next_state,axis=0))[0])
        #     # print('target time: ' + str(time() - t))


        #     target_f = self.model.predict(np.expand_dims(state,axis=0))
        #     target_f[0][action] = target
        #     # print('target_f time: ' + str(time() - t))
        #     if targets is not None:
        #         targets = np.append(targets,target_f,axis=0)
        #         states = np.append(states,np.expand_dims(state,axis=0),axis=0)
        #     else:
        #         # print(target_f)
        #         # print(target_f.shape)
        #         targets = target_f
        #         states = np.expand_dims(state,axis=0)

        # history = self.model.fit(np.expand_dims(state,axis=0), target_f, epochs=1, verbose=0)
        # loss = history.history['loss'][-1]

        # print('iteration time: ' + str(time() - t))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss


if __name__ == "__main__":
    env = WormWorldEnv(enable_render=True,world_size=(32,32))
    # nextaction = 'forward'
    # for n in range(1000):
    #     state, reward, done, info = env.step(action=nextaction)
    #     nextaction = 'forward'

    #     if np.mean(state[:5]) < np.mean(state[5:]):
    #         if np.random.random(1) < .3:
    #             if np.random.random(1) < .5:
    #                 nextaction = 'left'
    #             else:
    #                 nextaction = 'right'
    #     elif np.mean(state[:5]) == np.mean(state[5:]):
    #         if np.random.random(1) < .01:
    #             if np.random.random(1) < .5:
    #                 nextaction = 'left'
    #             else:
    #                 nextaction = 'right'

    #    # print(state)
    #    # print(reward)
    # exit()

    state_size = env.observation_space.shape[0]
    # action_size = len(env.action_space)
    action_size = 4
    agent = WormAgent(state_size, action_size)

    done = False
    batch_size = 32

    EPISODES = 5
    # http://pymedia.org/tut/src/make_video.py.html
    for e in range(EPISODES):
        state = env.reset()
        # state = np.reshape(state, [1, state_size])
        for tt in range(500):
            # env.render()
            t = time()
            action = agent.act(state)
            # print('action: ' + str(time() - t))
            next_state, reward, done, _ = env.step(action)
            # print('step: ' + str(time() - t))
            # reward = reward if not done else -10
            # next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            # print('remember: ' + str(time() - t))
            # print(reward)
            state = next_state

            if done:
                # print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
                print('episode: ' + str(e) + '/' + str(EPISODES) + ', score: ' + str(tt) + ', e: ' + str(agent.epsilon))
                break

            if len(agent.memory) > batch_size:

                loss = agent.replay(batch_size)
                # print('replay: ' + str(time() - t))
                # Logging training loss every 10 timesteps
                if tt % 10 == 0:
                    print('episode: ' + str(e) + '/' + str(EPISODES) + ', time: ' + str(tt) + ', loss: ' + str(loss))

        agent.save_model()


