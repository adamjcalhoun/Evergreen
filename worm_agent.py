import random
import gym
import numpy as np
from collections import deque

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import datetime

from envs.world_env import WormWorldEnv

from time import time
import argparse
import h5py
import pickle

import msvcrt

# For training the agent:
# https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym

# Multi-agent: https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/environment.py

# add in hierachical RL?
# https://towardsdatascience.com/advanced-reinforcement-learning-6d769f529eb3
class WormAgent:
    def __init__(self, state_size, action_size, load_only=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        if load_only:
            self.model = None
        else:
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

    def save_memory(self,dir_name=None):
        if dir_name is None:
            dir_name = ''

        with open(dir_name + 'memory_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.pkl','wb') as handle:
            pickle.dump(self.memory,handle)

        # with h5py.File(dir_name + 'memory_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.h5','w') as f:
        #     f.create_dataset('memory',data=np.array(self.memory))

    def load_model(self,dir_name=None,model_name=None):
        if dir_name is None:
            dir_name = ''

        if model_name == None:
            self.model = load_model(dir_name + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.h5')
        else:
            # self.model = load_model('/Users/adamjc/Dropbox/GitHub/Evergreen/' + model_name)
            self.model = load_model(dir_name + model_name)

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
                # print(next_state)
                # print(len(next_state))
                target = reward + self.gamma * np.amax(self.model.predict(np.expand_dims(np.array(next_state),axis=0))[0])
            # print('target time: ' + str(time() - t))

            # print('?')
            # print(state)
            # print(len(state))
            target_f = self.model.predict(np.expand_dims(state,axis=0))
            # print(target)
            # print(action)
            target_f[0][action] = target
            # print('target_f time: ' + str(time() - t))
            history = self.model.fit(np.expand_dims(state,axis=0), target_f, epochs=1, verbose=0)
            # print('history time: ' + str(time() - t))

            # print(history.history['loss'])
            loss += history.history['loss'][-1]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

def create_simple_environment(reward=''):
    env = WormWorldEnv(enable_render=args.show,world_size=(32,32),world_view_size=(1024,1024),reward_plan=reward)
    pid = env.add_odor_source(source_pos=(14,14),death_rate=0.01,diffusion_scale=2,emit_rate=0)

    # pid = env.add_odor_source(source_pos=(20,18),death_rate=0.01,diffusion_scale=1,emit_rate=1,plume_id=pid)
    # pid = env.add_circular_odor_source(source_pos=(20,18),plume_id=pid,radius=4,emit_rate=0.1)

    pid = env.add_square_odor_source(plume_id=pid,source_topleft=(10,18),source_bottomright=(24,20),emit_rate=0.1)
    env.add_vis_layer(layer_type='odor',pid=pid)
    env.set_odor_source_type(source_type='food',pid=pid)

    # tid = env.add_temp_gradient(source_pos=(14,14),fix_x=14,fix_y=None,tau=1,peak=22,trough=18)
    # env.set_agent_temp(mean_temp=18)
    return env

# def create_two_patch_environment(reward=''):
#     env = WormWorldEnv(enable_render=args.show,world_size=(128,128),world_view_size=(1024,1024),reward_plan=reward)
#     pid = env.add_odor_source(source_pos=(14,14),death_rate=0.1,diffusion_scale=2,emit_rate=0)

#     # pid = env.add_odor_source(source_pos=(20,18),death_rate=0.01,diffusion_scale=1,emit_rate=1,plume_id=pid)
#     # pid = env.add_circular_odor_source(source_pos=(20,18),plume_id=pid,radius=4,emit_rate=0.1)

#     pid = env.add_square_odor_source(plume_id=pid,source_topleft=(10,15),source_bottomright=(25,30),emit_rate=1)
#     env.add_vis_layer(layer_type='odor',pid=pid)
#     env.set_odor_source_type(source_type='food',pid=pid)

#     pid = env.add_square_odor_source(plume_id=pid,source_topleft=(60,60),source_bottomright=(75,75),emit_rate=3)
#     # env.add_vis_layer(layer_type='odor',pid=pid)  # having two of these makes the visualization look way better...
#     env.set_odor_source_type(source_type='food',pid=pid)

#     # tid = env.add_temp_gradient(source_pos=(14,14),fix_x=14,fix_y=None,tau=1,peak=22,trough=18)
#     # env.set_agent_temp(mean_temp=18)
#     return env

def create_two_patch_diffuse_environment(reward=''):
    env = WormWorldEnv(enable_render=args.show,world_size=(128,128),world_view_size=(1024,1024),reward_plan=reward)
    pid = env.add_odor_source(source_pos=(14,14),death_rate=0.1,diffusion_scale=10,emit_rate=0)

    # pid = env.add_odor_source(source_pos=(20,18),death_rate=0.01,diffusion_scale=1,emit_rate=1,plume_id=pid)
    # pid = env.add_circular_odor_source(source_pos=(20,18),plume_id=pid,radius=4,emit_rate=0.1)

    pid = env.add_square_odor_source(plume_id=pid,source_topleft=(10,15),source_bottomright=(25,30),emit_rate=1)
    env.add_vis_layer(layer_type='odor',pid=pid)
    env.set_odor_source_type(source_type='food',pid=pid)

    pid = env.add_square_odor_source(plume_id=pid,source_topleft=(60,60),source_bottomright=(75,75),emit_rate=3)
    # env.add_vis_layer(layer_type='odor',pid=pid)  # having two of these makes the visualization look way better...
    env.set_odor_source_type(source_type='food',pid=pid)

    # tid = env.add_temp_gradient(source_pos=(14,14),fix_x=14,fix_y=None,tau=1,peak=22,trough=18)
    # env.set_agent_temp(mean_temp=18)
    return env

def create_two_patch_dense_environment(reward='',radius=8,emit_rate1=1,emit_rate2=1):
    env = WormWorldEnv(enable_render=args.show,world_size=(128,128),world_view_size=(1024,1024),reward_plan=reward)
    pid = env.add_odor_source(source_pos=(14,14),death_rate=0.1,diffusion_scale=10,emit_rate=0)

    pid = env.add_square_odor_source(plume_id=pid,source_pos=(32,64),radius=radius,emit_rate=emit_rate1)
    env.add_vis_layer(layer_type='odor',pid=pid)
    env.set_odor_source_type(source_type='food',pid=pid)

    pid = env.add_square_odor_source(plume_id=pid,source_pos=(96,64),radius=radius,emit_rate=emit_rate2)
    env.set_odor_source_type(source_type='food',pid=pid)

    return env

def create_two_patch_environment(reward='',radius=8,emit_rate1=1,emit_rate2=1,coarseness=.1):
    env = WormWorldEnv(enable_render=args.show,world_size=(128,128),world_view_size=(1024,1024),reward_plan=reward)
    pid = env.add_odor_source(source_pos=(14,14),death_rate=0.1,diffusion_scale=10,emit_rate=0)

    # create patchy environment by placing elements in a loop?
    # or by creating a grid and "coarsening" it?
    pid = env.add_circular_odor_source(plume_id=pid,source_pos=(32,32),radius=15,emit_rate=emit_rate1,coarseness=coarseness)
    env.add_vis_layer(layer_type='odor',pid=pid)
    env.set_odor_source_type(source_type='food',pid=pid)

    pid = env.add_circular_odor_source(plume_id=pid,source_pos=(96,96),radius=15,emit_rate=emit_rate2,coarseness=coarseness)
    env.set_odor_source_type(source_type='food',pid=pid)

    return env

def create_large_environment(reward=''):
    # 512x512? or larger?
    env = WormWorldEnv(enable_render=args.show,world_size=(512,512),world_view_size=(1024,1024),reward_plan=reward)
    pid = env.add_odor_source(source_pos=(14,14),death_rate=0.01,diffusion_scale=2,emit_rate=0)

    pid = env.add_square_odor_source(plume_id=pid,source_topleft=(10,18),source_bottomright=(24,20),emit_rate=0.1)
    pid = env.add_square_odor_source(plume_id=pid,source_topleft=(400,18),source_bottomright=(424,20),emit_rate=1)
    pid = env.add_square_odor_source(plume_id=pid,source_topleft=(10,408),source_bottomright=(24,420),emit_rate=5)
    pid = env.add_square_odor_source(plume_id=pid,source_topleft=(410,418),source_bottomright=(424,420),emit_rate=0.1)

    pid = env.add_square_odor_source(plume_id=pid,source_topleft=(190,190),source_bottomright=(200,200),emit_rate=1)
    
    env.add_vis_layer(layer_type='odor',pid=pid)
    env.set_odor_source_type(source_type='food',pid=pid)

    # tid = env.add_temp_gradient(source_pos=(14,14),fix_x=14,fix_y=None,tau=1,peak=22,trough=18)
    # env.set_agent_temp(mean_temp=18)
    return env

# sbatch submitGeneric.sbatch "-d twopatch_rfh_5/ -rf -ih -a 5 -t 10000 -e 1000 -w two_patch"
# sbatch submitGeneric.sbatch "-d twopatch_diffuse_rfh_5/ -rf -ih -a 5 -t 10000 -e 1000 -w two_patch_diffuse"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='worm_agent')
    parser.add_argument('-s', '--show', action='store_true', help='Show pygame visualization')
    parser.add_argument('--load_model', type=str, help='Load a pre-trained model to use as the agent', default='')
    parser.add_argument('-d', '--dir', type=str, help='Choose where to save data to', default='')

    parser.add_argument('--simulate', help='Simulate only', action='store_true')

    parser.add_argument('-ro', '--reward_odor', action='store_true', help='Reward on odor')
    parser.add_argument('-rt', '--reward_temp', action='store_true', help='Reward on temp')
    parser.add_argument('-rf', '--reward_food', action='store_true', help='Reward on food')
    parser.add_argument('-re', '--reward_eggs', action='store_true', help='Reward for egg laying')
    parser.add_argument('-ih', '--reward_hunger', action='store_true', help='Have an internal hunger variable (add to state)')

    parser.add_argument('-w', '--world', type=str, help='Pre-defined world to run', default='')
    parser.add_argument('-a', '--num_agents', type=int, help='Number of agents to simulate', default=1)

    parser.add_argument('-t', '--num_timesteps', type=int, help='Number of time steps per epoch', default=1000)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs', default=1)
    
    args = parser.parse_args()

    reward_string = ''
    if args.reward_odor:
        reward_string += 'odor,'
    if args.reward_temp:
        reward_string += 'temp,'
    if args.reward_food:
        reward_string += 'food,'
    if args.reward_eggs:
        reward_string += 'eggs,'
    if args.reward_hunger:
        reward_string += 'hunger,'

    num_agents = args.num_agents
    num_timesteps = args.num_timesteps
    if args.world == '':
        # env = WormWorldEnv(enable_render=True,world_size=(32,32),world_view_size=(512,512))
        # env = create_simple_environment(reward=reward_string)
        env = create_two_patch_environment(reward=reward_string,coarseness=0.1)
        # might need to make the patches closer in space
        # env = create_large_environment(reward=reward_string)
    else:
        env = locals()['create_' + args.world + '_environment'](reward=reward_string)

    pid = env.set_num_agents(num_agents=num_agents)
    # self-odor: ADD THIS AS A FLAG
    # env.add_vis_layer(layer_type='odor',pid=pid)


    # (self,temp_id=None,source_pos=(0,0),fix_x=None,fix_y=None,tau=1,peak=25,trough=18)
    
    # env.add_vis_layer(layer_type='temp',pid=tid)

    env.fix_environment() # save the environment for when we need to reset it

    state_size = env.observation_space.shape[0]
    # action_size = len(env.action_space)
    action_size = 4
    agent = WormAgent(state_size, action_size)
    if args.load_model != '':
        agent.load_model(model_name=args.load_model)

    done = False
    batch_size = 32

    # if args.simulate:
    #     EPISODES = 1
    # else:
    #     EPISODES = 20

    abort_program = False
    # http://pymedia.org/tut/src/make_video.py.html
    for e in range(args.epochs):
        state = env.reset()
        # state = np.reshape(state, [1, state_size])
        for tt in range(num_timesteps):
            if msvcrt.kbhit():
                print('detected enter, exiting!')
                abort_program = True
                break

            # env.render()
            t = time()
            for aa in range(num_agents):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)

                agent.remember(state, action, reward, next_state, done)
                state = next_state

            if done and not args.simulate:
                print('episode: ' + str(e) + '/' + str(args.epochs) + ', score: ' + str(tt) + ', e: ' + str(agent.epsilon))
                break

            if len(agent.memory) > batch_size and not args.simulate:
                # replay is slow...

                loss = agent.replay(batch_size)
                if tt % 10 == 0:
                    print('episode: ' + str(e) + '/' + str(args.epochs) + ', time: ' + str(tt) + ', loss: ' + str(loss))

        if not args.simulate:
            agent.save_model(dir_name=args.dir)
        else:
            agent.save_memory(dir_name=args.dir)

        if abort_program:
            break



