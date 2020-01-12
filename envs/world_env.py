import numpy as np
import copy

import gym
from gym import error, spaces, utils
from gym.utils import seeding
# from gym_maze.envs.maze_view_2d import MazeView2D
from envs.world_view_2d import WorldView2D

class WormWorldEnv(gym.Env):
    # metadata = {
    #     "render.modes": ["human", "rgb_array"],
    # }

    # https://stackoverflow.com/questions/44404281/openai-gym-understanding-action-space-notation-spaces-box
    ACTION = ['forward','backward','left','right']
    # additional action: stay, lay eggs?

    def __init__(self, world_file=None, world_size=(512,512), world_view_size=None, enable_render=False, reward_plan='', /
        hunger=False, lay_eggs=False):

        self.viewer = None
        self.enable_render = enable_render
        self.__vis_layers = []
        self.game_over = False
        self.position = None
        self.hunger = 0
        self.hunger_step = 0.01 # these need to be passable parameters
        self.max_hunger = 1
        self.odor_history_length = 10
        self.odor_history = np.zeros(self.odor_history_length)

        self.temp_history_length = 10
        self.temp_history = np.zeros(self.temp_history_length)

        self.has_hunger = True

        self.eggs = []
        self.egg_delay = 5
        self.egg_reward_per_food = 1
        self.egg_reward_radius = 5

        self.reward_plan = np.zeros((4,))
        for reward_name in reward_plan.split(','):
            if reward_name == 'odor':
                self.reward_plan[0] = 1
            elif reward_name == 'temp':
                self.reward_plan[1] = 1
            elif reward_name == 'food':
                self.reward_plan[2] = 1
            elif reward_name == 'egg':
                pass

        # need to fix this...
        self.observation_space = spaces.Box(np.zeros(self.odor_history_length + self.temp_history_length + self.has_hunger,),np.ones(self.odor_history_length + self.temp_history_length + self.has_hunger,))
        self.action_space = spaces.Discrete(4)

        self.__world = World(world_size=world_size)
        self.__world.add_agent(location=[10,10])

        self.__world_save = None

        # initial condition
        self.state = None
        self.steps_beyond_done = None
        self.world = np.zeros(world_size)

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()

        if self.enable_render:
            self.viewer = WorldView2D(world_size=world_size,world_view_size=world_view_size)

    def configure(self):
        pass

    def __del__(self):
        if self.enable_render is True:
            self.viewer.quit_game()

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def step(self, action):
        if isinstance(action, str):
            newstate = self.__world.agent_action(action)
        else:
            newstate = self.__world.agent_action(self.ACTION[action])

        self.__world.step()
        if self.enable_render:
            viewlayers = [self.__world.get_env_values(layer_type=layer[0],pid=layer[1]) for layer in self.__vis_layers]

            self.viewer.view_update(agent=self.__world.get_agent_location(),viewlayers=viewlayers,agent_angle=self.__world.agent.get_angled())

        self.update_history(newstate)

        if self.has_hunger:
            self.hunger -= self.hunger_step + newstate[2]
            self.hunger = min(hunger,self.max_hunger)

        reward = 0
        if self.reward_plan[0] == 1:
            reward += np.exp(newstate[0])

        # pos = self.__world.get_agent_location()
        if self.reward_plan[1] == 1:
            reward -= np.abs(self.__world.agent.get_agent_temp() - newstate[1])

        if self.reward_plan[2] == 1:
            reward += newstate[2]   # reward for eating food

            # will need to control for whether the hunger variable can be indicative of food even in the absence
            # of eating/internal hunger:
            # so don't always reward it
            if self.has_hunger:
                if self.hunger < 0:
                    reward += self.hunger

        # # for laying eggs
        # if self.reward_plan[3] == 1:
        #     for egg in self.eggs:
        #         egg[1] -= 1 # decrement the egg delay by one time step

        #         if egg[1] < 0:      # less than 0 because we decrement immediately
        #             reward += self.egg_reward_radius * some_matrix  # this part isn't immediately trivial
        #     self.eggs = [good_egg for good_egg in self.eggs if good_egg[1] < 0]


        # if np.array_equal(self.worm_view.robot, self.worm_view.goal):
        #     reward = 1
        #     done = True
        # else:
        #     reward = -0.1/(self.worm_size[0]*self.worm_size[1])
        #     done = False
        # reward = 0
        done = False

        # print('newstate: ' + str(newstate) + ' ... reward: ' + str(reward))

        # print(np.append(self.odor_history,self.temp_history))
        self.state = list(np.append(self.odor_history,self.temp_history))
        if self.has_hunger:
            self.state.append(self.hunger)
        # print(len(self.state))

        info = {}

        return self.state, reward, done, info

    def add_odor_source(self,source_pos=(0,0),death_rate=0,diffusion_scale=0,emit_rate=0,plume_id=None):
        return self.__world.add_odor_source(source_pos=source_pos,death_rate=death_rate,diffusion_scale=diffusion_scale,emit_rate=emit_rate,plume_id=plume_id)

    def add_circular_odor_source(self,source_pos=(0,0),plume_id=0,radius=0,emit_rate=0):
        return self.__world.add_circular_odor_source(source_pos=source_pos,plume_id=plume_id,radius=radius,emit_rate=emit_rate)

    def add_square_odor_source(self,plume_id=0,source_topleft=None,source_bottomright=None,emit_rate=0):
        return self.__world.add_square_odor_source(plume_id=plume_id,source_topleft=source_topleft,source_bottomright=source_bottomright,emit_rate=emit_rate)

    def set_odor_source_type(self,source_type='none',pid=0):
        self.__world.set_odor_source_type(source_type=source_type,pid=pid)

    def set_agent_temp(self,mean_temp=20):
        self.__world.agent.set_agent_temp(mean_temp=mean_temp)

    def add_temp_gradient(self,temp_id=None,source_pos=(0,0),fix_x=None,fix_y=None,tau=1,peak=25,trough=18):
        return self.__world.add_temp_gradient(temp_id=temp_id,source_pos=source_pos,fix_x=fix_x,fix_y=fix_y,tau=tau,peak=peak,trough=trough)

    def update_history(self,newstate):
        self.odor_history[1:] = self.odor_history[:-1]
        self.odor_history[0] = newstate[0]
        self.temp_history[1:] = self.temp_history[:-1]
        self.temp_history[0] = newstate[1]

    def reset(self):
        # self.maze_view.reset_robot()
        # create environment
        # choose location for worm
        if self.__world_save is not None:
            self.__world = self.__world_save

        self.__world.set_agent_location([10,10])
        self.state = np.zeros(self.odor_history_length+self.temp_history_length,)
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def fix_environment(self):
        print('saved!')
        self.__world_save = copy.deepcopy(self.__world)
        print(self.__world_save)

    def is_game_over(self):
        # return self.maze_view.game_over
        return self.game_over

    def add_vis_layer(self,layer_type='none',pid=-1):
        if layer_type == 'odor' or layer_type == 'temp':
            self.__vis_layers.append((layer_type,pid))


class World:
    def __init__(self, world_cells=None, world_size=(512,512), num_odor_sources=0):

        # maze member variables
        self.world_size = world_size
        self.odor_sources = []
        self.odor_type = []
        self.temp_layers = []
        self.agent = None


        # maze's configuration parameters
        if not (isinstance(world_size, (list, tuple)) and len(world_size) == 2):
            raise ValueError("world_size must be a tuple: (width, height).")
        self.world_size = world_size

        self._generate_world(num_odor_sources=num_odor_sources)

    def _generate_world(self,num_odor_sources):

        # list of all cell locations
        self.world_cells = np.zeros(self.world_size, dtype=int)

        # Initializing constants and variables needed for maze generation
        for pid in list(range(num_odor_sources)):
            plume = OdorPlume(world_size=self.world_size,plume_id=pid,death_rate=0.001,diffusion_scale=3)
            for n in range(num_odor_sources):
                plume.add_source(source_pos=(np.random.random(1)*self.world_size[0],np.random.random(1)*self.world_size[1]),emit_rate=np.random.random(1)*5)

            plume.step()

            self.odor_sources.append(plume)

    def add_odor_source(self,source_pos=(0,0),death_rate=0,diffusion_scale=0,emit_rate=0,plume_id=None):
        if plume_id is None:
            plume_id = len(self.odor_sources)
            plume = OdorPlume(world_size=self.world_size,plume_id=plume_id,death_rate=death_rate,diffusion_scale=diffusion_scale)
            self.odor_type.append('none')
            self.odor_sources.append(plume)
        else:
            plume = self.odor_sources[plume_id]

        plume.add_source(source_pos=source_pos,emit_rate=emit_rate)

        return plume_id

    def add_circular_odor_source(self,source_pos=(0,0),plume_id=0,radius=0,emit_rate=0):
        plume = self.odor_sources[plume_id]
        return plume.add_circular_source(source_pos=source_pos,radius=radius,emit_rate=emit_rate)

    def add_square_odor_source(self,plume_id=0,source_topleft=None,source_bottomright=None,emit_rate=0):
        plume = self.odor_sources[plume_id]
        return plume.add_square_source(source_topleft=source_topleft,source_bottomright=source_bottomright,emit_rate=emit_rate)

    def set_odor_source_type(self,source_type='none',pid=0):
        self.odor_type[pid] = source_type

    def get_odor_source_type(self,pid):
        return self.odor_type[pid]

    def add_temp_gradient(self,temp_id=None,source_pos=(0,0),fix_x=None,fix_y=None,tau=1,peak=25,trough=18):
        if temp_id is None:
            temp_id = len(self.temp_layers)
            temp = TemperatureGradient(world_size=self.world_size)
            self.temp_layers.append(temp)
        else:
            temp = self.temp_layers[temp_id]

        temp.make_gaussian_peak(source_pos=source_pos,fix_x=fix_x,fix_y=fix_y,tau=tau,peak=peak,trough=trough)

        return temp_id

    def add_agent(self,location=None):
        if self.agent is None:
            self.agent = Agent(location=location)

    def agent_action(self,action=None):
        if action == 'left':
            self.agent.rotate(np.pi/2)
        elif action == 'right':
            self.agent.rotate(-np.pi/2)

        if action == 'forward' or action == 'left' or action == 'right':
            self.agent.move(forward_backward=1)
            # check for edge effects
            location = self.agent.get_location()
            if location[0] >= self.world_size[0]:
                location[0] = 0
            if location[1] >= self.world_size[1]:
                location[1] = 0
            if location[0] < 0:
                location[0] = self.world_size[0]-1
            if location[1] < 0:
                location[1] = self.world_size[1]-1
            self.agent.set_location(location)

        elif action == 'backward':
            self.agent.move(forward_backward=-1)
            location = self.agent.get_location()
            if location[0] >= self.world_size[0]:
                location[0] = 0
            if location[1] >= self.world_size[1]:
                location[1] = 0
            if location[0] < 0:
                location[0] = self.world_size[0]-1
            if location[1] < 0:
                location[1] = self.world_size[1]-1
            self.agent.set_location(location)

        if action == 'lay_egg':
            # cost to lay egg that is rewarded based on food near animal at timestep t+T
            self.eggs.append([self.agent.get_location(),self.egg_delay])


        state = []
        amt_decay =0
        (x,y) = self.agent.get_location()
        for odor_num,odor in enumerate(self.odor_sources):
            state.append(odor.get_odor_value(x,y))
            # if xxxxx contains 'animal'
            if self.get_odor_source_type(pid=odor_num) == 'food':
                # state.append(odor.get_odor_emit(x=x,y=y))
                amt_decay += odor.decay_plume(x=x,y=y,decay_amount=1)

            odor.consume_plume(x=x,y=y)

        for temp_num,temp in enumerate(self.temp_layers):
            state.append(temp.get_temp_value(x,y))

        state.append(amt_decay)

        return state

    def get_agent_location(self):
        return self.agent.get_location()

    def set_agent_location(self,location):
        return self.agent.set_location(location)

    def get_odor_plumes(self):
        return self.odor_sources

    def get_odor_values(self,pid=None):
        if pid is not None:
            return self.odor_sources[pid].get_full_odor()
        else:
            return [odor.get_full_odor() for odor in self.odor_sources]

    def get_temp_values(self,pid=None):
        if pid is not None:
            return self.temp_layers[pid].get_full_temp()
        else:
            return [odor.get_full_temp() for temp in self.temp_layers]

    def get_env_values(self,layer_type='none',pid=None):
        if layer_type == 'odor':
            return self.get_odor_values(pid=pid)
        elif layer_type == 'temp':
            return self.get_temp_values(pid=pid)

    def step(self):
        for odor in self.odor_sources:
            odor.step()

        self.agent.step()

    @property
    def WORLD_W(self):
        return int(self.world_size[0])

    @property
    def WORLD_H(self):
        return int(self.world_size[1])



class Agent:
    def __init__(self,location=[10,10],direction=0):
        self.__location = location
        self.__direction = direction
        self.__mean_temp = 0

    def update_location(self,location=None):
        self.__location = location

    def get_location(self):
        return self.__location

    def rotate(self,angle):
        self.__direction += angle

    def move(self,forward_backward=1):
        self.__location[0] += int(np.cos(self.__direction)*forward_backward)
        self.__location[1] += int(np.sin(self.__direction)*forward_backward)

    def set_agent_temp(self,mean_temp=20):
        self.__mean_temp = mean_temp

    def get_agent_temp(self):
        return self.__mean_temp

    def set_location(self,location):
        self.__location = location

    def get_angle(self):
        return self.__direction

    def get_angled(self):
        return np.rad2deg(self.__direction)

    def step(self):
        # place-holder for when we want to do more things (hunger etc)
        pass


class OdorPlume:

    def __init__(self, world_size=(512,512), plume_id=None, death_rate=0.001, decay_rate=0,
                diffusion_scale=1, min_concentration=0.00001, source_consumable=False):

        self.plume_id = plume_id

        self.odor = np.zeros(world_size)
        self.world_size = world_size

        self.death_rate = death_rate
        self.decay_rate = []
        self.diffusion_scale = diffusion_scale

        self.min_concentration = min_concentration
        self.source_consumable = source_consumable

        self.source_pos = []
        self.emit_rate = []
        self.num_sources = 0

        self.source_agent = []


    def add_source(self,source_pos=None,emit_rate=None, source_agent=None):
        if (source_pos is None and source_agent is None) or (source_pos is not None and source_agent is not None):
            print('error! please specify either the source position or the agent, but not both')
            return None

        if source_pos is not None and self.source_agent != []:
            print('error! cannot add new immobile source when this source comes from an agent')

        if source_pos is not None:
            self.source_pos.append((int(source_pos[0]),int(source_pos[1])))
        if source_agent is not None:
            self.source_agent.append(source_agent)
            self.source_pos.append((source_agent.get_location()))

        self.emit_rate.append(emit_rate)
        self.num_sources += 1


    def add_circular_source(self,source_pos=None,radius=None,emit_rate=None):
        x = np.arange(0,self.world_size[0])
        y = np.arange(0,self.world_size[1])
        xx,yy = np.meshgrid(x,y)
        circle = np.sqrt((xx - source_pos[0])**2 + (yy - source_pos[1])**2) < radius

        inds = np.where(circle)

        for (vv,ww) in zip(inds[0],inds[1]):
            self.source_pos.append((int(vv),int(ww)))
            self.emit_rate.append(emit_rate)

        self.num_sources += np.sum(circle)

    def add_square_source(self,source_topleft=None,source_bottomright=None,emit_rate=None):
        square = np.zeros((self.world_size))
        square[source_topleft[0]:source_bottomright[0],source_topleft[1]:source_bottomright[1]] = 1

        inds = np.where(square == 1)

        for (vv,ww) in zip(inds[0],inds[1]):
            self.source_pos.append((int(vv),int(ww)))
            self.emit_rate.append(emit_rate)

            self.num_sources += 1

        return self.plume_id

    def step(self):
        if self.source_agent is not None:
            self.source_pos = [agent.get_pos for agent in self.source_agent]

        world_diffusion = np.random.random([self.world_size[0],self.world_size[1],4])
        world_diffusion /= np.tile(np.expand_dims(np.sum(world_diffusion,axis=2),axis=-1),(1,1,4))

        # probably a more pythonic way to do this...
        for snum in list(range(self.num_sources)):
            self.odor[self.source_pos[snum][0],self.source_pos[snum][1]] += self.emit_rate[snum]

        # assume everything moves diffusion_scale steps in cardinal directions (for simplicity)
        # this one seems to go diagonally upward for some reason...
        # self.odor = np.append(self.odor[self.diffusion_scale:,:]*world_diffusion[self.diffusion_scale:,:,0], self.odor[-self.diffusion_scale:,:]*world_diffusion[-self.diffusion_scale:,:,0],axis=0) + np.append(self.odor[:-self.diffusion_scale,:]*world_diffusion[:-self.diffusion_scale,:,1], self.odor[:self.diffusion_scale,:]*world_diffusion[:self.diffusion_scale,:,1],axis=0) + np.append(self.odor[:,self.diffusion_scale:]*world_diffusion[:,self.diffusion_scale:,2], self.odor[:,-self.diffusion_scale:]*world_diffusion[:,-self.diffusion_scale:,2],axis=1) + np.append(self.odor[:,:-self.diffusion_scale]*world_diffusion[:,:-self.diffusion_scale,3], self.odor[:,:self.diffusion_scale]*world_diffusion[:,:self.diffusion_scale,3],axis=1)

        thisscale = int(np.random.random(1)*(self.diffusion_scale-1))+1
        self.odor = np.append(self.odor[thisscale:,:]*world_diffusion[thisscale:,:,0], self.odor[:thisscale,:]*world_diffusion[:thisscale,:,0],axis=0) + np.append(self.odor[-thisscale:,:]*world_diffusion[-thisscale:,:,1], self.odor[:-thisscale,:]*world_diffusion[:-thisscale,:,1],axis=0) + np.append(self.odor[:,thisscale:]*world_diffusion[:,thisscale:,2], self.odor[:,:thisscale]*world_diffusion[:,:thisscale,2],axis=1) + np.append(self.odor[:,-thisscale:]*world_diffusion[:,-thisscale:,3], self.odor[:,:-thisscale]*world_diffusion[:,:-thisscale,3],axis=1)
        # self.odor = np.append(self.odor[self.diffusion_scale:,:]*world_diffusion[self.diffusion_scale:,:,0], self.odor[:self.diffusion_scale,:]*world_diffusion[:self.diffusion_scale,:,0],axis=0) + np.append(self.odor[-self.diffusion_scale:,:]*world_diffusion[-self.diffusion_scale:,:,1], self.odor[:-self.diffusion_scale,:]*world_diffusion[:-self.diffusion_scale,:,1],axis=0) + np.append(self.odor[:,self.diffusion_scale:]*world_diffusion[:,self.diffusion_scale:,2], self.odor[:,:self.diffusion_scale]*world_diffusion[:,:self.diffusion_scale,2],axis=1) + np.append(self.odor[:,-self.diffusion_scale:]*world_diffusion[:,-self.diffusion_scale:,3], self.odor[:,:-self.diffusion_scale]*world_diffusion[:,:-self.diffusion_scale,3],axis=1)

        # np.append(x[1:,:]*.25, x[:1,:]*.25,axis=0) + np.append(x[-1:,:]*.25, x[:-1,:]*.25,axis=0) + np.append(x[:,1:]*.25, x[:,:1]*.25,axis=1) + np.append(x[:,-1:]*.25, x[:,:-1]*.25,axis=1)

        # or lose random %? or divide?
        self.odor -= self.death_rate
        self.odor[self.odor < self.min_concentration] = 0
        # print((len(self.emit_rate),len(self.source_pos)))

        
    def get_odor_value(self,x,y):
        return self.odor[int(x),int(y)]

    def get_odor_emit(self,x,y):
        # print([i for i,val in enumerate(self.source_pos) if val == (x,y)])
        plume_number = [i for i,val in enumerate(self.source_pos) if val == (x,y)]
        if len(plume_number) > 0:
            plume_number = plume_number[0]
            return self.emit_rate[plume_number]
        else:
            return 0

    def get_full_odor(self):
        return self.odor

    def get_sources(self):
        return self.source_pos

    def get_plume_id(self):
        return self.plume_id

    def consume_plume(self,x,y,pct=1):
        x = int(x)
        y = int(y)
        consume = (1-pct)*self.odor[x,y]
        self.odor[x,y] = consume

        return consume

    def decay_plume(self,plume_number=None,decay_amount=0,x=-1,y=-1):
        # print((plume_number,x,y))
        if x != -1 and y != -1:
            if (x,y) not in self.source_pos:
                return 0
            else:
                plume_number = [i for i,val in enumerate(self.source_pos) if val == (x,y)][0]

        # print([i for i,val in enumerate(self.source_pos) if val == (x,y)])
        # print([i for i,val in enumerate(self.source_pos) if val == (x,y)])
        # print(len(self.source_pos))
        # print(plume_number)
        # print('decaying ' + str(plume_number) + ' by ' + str(decay_amount))
        old_amt = self.emit_rate[plume_number]
        amt_decay = max(self.emit_rate[plume_number]-decay_amount,0)
        self.emit_rate[plume_number] = amt_decay

        return (old_amt - amt_decay)

    def decay_all_plumes(self,decay_amount=0):
        # print('decay all plumes?')
        amt_decay = 0
        for pnum in list(range(self.num_sources)):
            amt_decay += decay_plume(plume_number=pnum, decay_amount=decay_amount)

        return amt_decay


class TemperatureGradient:

    def __init__(self, world_size=(512,512), gradient=None):

        self.__temp = np.zeros(world_size)
        self.__world_size = world_size

        if gradient is not None:
            self.__temp = gradient

    def make_gaussian_peak(self,source_pos=(50,50),fix_x=None,fix_y=None,tau=1,peak=25,trough=18):
        if fix_x is not None:
            x = np.ones(self.__world_size[1])*fix_x
        else:
            x = np.arange(0,self.__world_size[0])
            
        if fix_y is not None:
            y = np.ones(self.__world_size[1])*fix_y
        else:
            y = np.arange(0,self.__world_size[0])
            

        xx,yy = np.meshgrid(x,y)
        self.__temp = np.exp(-((xx - source_pos[0])**2 + (yy - source_pos[1])**2) / tau) * (peak-trough) + trough

    def set_gradient(self,gradient=None):
        self.__temp = gradient

    def get_temp_value(self,x,y):
        return self.__temp[x,y]

    def get_full_temp(self):
        return self.__temp


if __name__ == "__main__":

    world = WorldView2D(world_size=(500, 500))
    for t in range(200):
        world.update()
    input("Enter any key to quit.")