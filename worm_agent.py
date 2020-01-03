import random
import gym
import numpy as np
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import datetime

from envs.world_env import WormWorldEnv

from time import time
import argparse

# For training the agent:
# https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym

class WormAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()


    def _build_model(self,model_type='dense'):
        # Neural Net for Deep-Q learning Model
        if model_type == 'dense':
            model = Sequential()
            model.add(Dense(12, input_dim=self.state_size, activation='relu'))
            model.add(Dense(12, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        elif model_type == 'deeperdense':
            model = Sequential()
            model.add(Dense(12, input_dim=self.state_size, activation='relu'))
            model.add(Dense(12, activation='relu'))
            model.add(Dense(12, activation='relu'))
            model.add(Dense(12, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        elif model_type == 'dense_batchnorm':
            model = Sequential()
            model.add(Dense(12, input_dim=self.state_size, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(12, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        elif model_type == 'dense_batchnorm2':
            model = Sequential()
            model.add(BatchNormalization(input_dim=self.state_size))
            model.add(Dense(12, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(12, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        elif model_type == 'LSTM':
            pass
        elif model_type == 'dense_feedback':
            pass

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
        # print(self.memory)

        for state, action, reward, next_state, done in minibatch:
            t = time()
            target = reward
            # print('training')

            # print((len(state),state))
            # print(np.array(next_state).shape)
            # print(self.model.predict(np.expand_dims(next_state,axis=0)))
            # print((reward,np.amax(self.model.predict(np.expand_dims(next_state,axis=0))[0]),self.model.predict(np.expand_dims(next_state,axis=0))[0]))
            # print('!!!!')

            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.expand_dims(next_state,axis=0))[0])
            # print('target time: ' + str(time() - t))

            target_f = self.model.predict(np.expand_dims(state,axis=0))
            # print(target)
            # print(action)
            target_f[0][action] = target
            # print('target_f time: ' + str(time() - t))
            history = self.model.fit(np.expand_dims(state,axis=0), target_f, epochs=1, verbose=0)
            print('history time: ' + str(time() - t))

            # print(history.history['loss'])
            loss += history.history['loss'][-1]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='worm_agent')
    parser.add_argument('-s', '--show', action='store_true', help='Show pygame visualization')
    args = parser.parse_args()

    # env = WormWorldEnv(enable_render=True,world_size=(32,32),world_view_size=(512,512))
    env = WormWorldEnv(enable_render=args.show,world_size=(32,32),world_view_size=(1024,1024))
    pid = env.add_odor_source(source_pos=(14,14),death_rate=0.01,diffusion_scale=2,emit_rate=0)
    # pid = env.add_odor_source(source_pos=(20,18),death_rate=0.01,diffusion_scale=1,emit_rate=1,plume_id=pid)
    # pid = env.add_circular_odor_source(source_pos=(20,18),plume_id=pid,radius=4,emit_rate=0.1)
    pid = env.add_square_odor_source(plume_id=pid,source_topleft=(10,18),source_bottomright=(24,20),emit_rate=0.1)
    env.add_vis_layer(layer_type='odor',pid=pid)
    env.set_odor_source_type(source_type='food',pid=pid)



    # (self,temp_id=None,source_pos=(0,0),fix_x=None,fix_y=None,tau=1,peak=25,trough=18)
    tid = env.add_temp_gradient(source_pos=(14,14),fix_x=14,fix_y=None,tau=1,peak=22,trough=18)
    env.set_agent_temp(mean_temp=18)
    # env.add_vis_layer(layer_type='temp',pid=tid)

    env.fix_environment() # save the environment for when we need to reset it

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
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print('episode: ' + str(e) + '/' + str(EPISODES) + ', score: ' + str(tt) + ', e: ' + str(agent.epsilon))
                break

            if len(agent.memory) > batch_size:

                loss = agent.replay(batch_size)
                if tt % 10 == 0:
                    print('episode: ' + str(e) + '/' + str(EPISODES) + ', time: ' + str(tt) + ', loss: ' + str(loss))

        agent.save_model()


