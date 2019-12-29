import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
# from gym_maze.envs.maze_view_2d import MazeView2D
from envs.world_view_2d import WorldView2D

class WormWorldEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    # https://stackoverflow.com/questions/44404281/openai-gym-understanding-action-space-notation-spaces-box
    ACTION = ['forward','backward','left','right']

    def __init__(self, world_file=None, world_size=(512,512), world_view_size=None, enable_render=False):

        self.viewer = None
        self.enable_render = enable_render
        self.game_over = False
        self.position = None
        self.hunger = 0
        self.hunger_step = 0.01 # these need to be passable parameters
        self.odor_history_length = 10
        self.odor_history = np.zeros(self.odor_history_length)

        # need to fix this...
        self.observation_space = spaces.Box(np.zeros(self.odor_history_length,),np.ones(self.odor_history_length,))
        self.action_space = spaces.Discrete(4)

        self.__world = World(world_size=world_size)
        self.__world.add_agent(location=[10,10])

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
            self.viewer.view_update(agent=self.__world.get_agent_location(),viewlayers=self.__world.get_odor_values(pid=0))

        self.update_odor_history(newstate)

        reward = np.exp(newstate)
        # if np.array_equal(self.worm_view.robot, self.worm_view.goal):
        #     reward = 1
        #     done = True
        # else:
        #     reward = -0.1/(self.worm_size[0]*self.worm_size[1])
        #     done = False
        # reward = 0
        done = False

        self.state = list(self.odor_history)

        info = {}

        return self.state, reward, done, info

    def add_odor_sources(self,source):
        pass

    def add_temp(self,temp):
        pass

    def update_odor_history(self,newstate):
        self.odor_history[1:] = self.odor_history[:-1]
        self.odor_history[0] = newstate[0]

    def reset(self):
        # self.maze_view.reset_robot()
        # create environment
        # choose location for worm
        self.__world.set_agent_location([10,10])
        self.state = np.zeros(10,)
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        # return self.maze_view.game_over
        return self.game_over


class World:

    COMPASS = {
        "N": (0, -1),
        "E": (1, 0),
        "S": (0, 1),
        "W": (-1, 0)
    }

    def __init__(self, world_cells=None, world_size=(512,512), num_odor_sources=(1)):

        # maze member variables
        self.world_size = world_size
        self.odor_sources = []
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
                # plume.add_source(source_pos=(np.random.random(1)*self.world_size[0],np.random.random(1)*self.world_size[1]),emit_rate=np.random.random(1)*5)
                plume.add_source(source_pos=(16,16),emit_rate=1)
                # plume.add_source(source_pos=(30,16),emit_rate=2)
                # plume.add_source(source_pos=(16,30),emit_rate=3)
                # plume.add_source(source_pos=(100,100),emit_rate=5)

            plume.step()

            self.odor_sources.append(plume)
        

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

        state = []
        for odor in self.odor_sources:
            (x,y) = self.agent.get_location()
            state.append(odor.get_odor_value(x,y))
            odor.consume_plume(x,y)

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

    def update_location(self,location=None):
        self.__location = location

    def get_location(self):
        return self.__location

    def rotate(self,angle):
        self.__direction += angle

    def move(self,forward_backward=1):
        self.__location[0] += int(np.cos(self.__direction)*forward_backward)
        self.__location[1] += int(np.sin(self.__direction)*forward_backward)

    def set_location(self,location):
        self.__location = location

    def step(self):
        # place-holder for when we want to do more things (hunger etc)
        pass


class OdorPlume:

    def __init__(self, world_size=(512,512), plume_id=None, death_rate=0.001, 
                diffusion_scale=1, min_concentration=0.00001, source_consumable=False):

        self.plume_id = plume_id

        self.odor = np.zeros(world_size)
        self.world_size = world_size

        self.death_rate = death_rate
        self.diffusion_scale = diffusion_scale

        self.min_concentration = min_concentration
        self.source_consumable = source_consumable

        self.source_pos = []
        self.emit_rate = []
        self.num_sources = 0


    def add_source(self,source_pos=None,emit_rate=None):
        self.source_pos.append((int(source_pos[0]),int(source_pos[1])))
        self.emit_rate.append(emit_rate)
        self.num_sources += 1

    def add_circular_source(self,source_pos=None,radius=None,emit_rate=None):
        x = np.arange(0,self.__world_size[0])
        y = np.arange(0,self.__world_size[1])
        xx,yy = np.meshgrid(x,y)
        circle = np.sqrt((xx - source_pos[0])**2 + (yy - source_pos[1])**2) < radius

        inds = np.where(circle)

        for (vv,ww) in zip(inds[0],inds[1]):
            self.source_pos.append((int(vv),int(ww)))
            self.emit_rate.append(emit_rate)

        self.num_sources += np.sum(circle)

    def add_square_source(self,source_topleft=None,source_bottomright=None,emit_rate=None):
        square = np.zeros((self.__world_size))
        square[source_topleft[0]:source_bottomright[0],source_topleft[1]:source_bottomright[1]]

        inds = np.where(square == 1)

        for (vv,ww) in zip(inds[0],inds[1]):
            self.source_pos.append((int(vv),int(ww)))
            self.emit_rate.append(emit_rate)

        self.num_sources += 1

    def step(self):
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

        
    def get_odor_value(self,x,y):
        return self.odor[int(x),int(y)]

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
        if x != -1 and y != -1:
            if (x,y) not in self.source_pos:
                return 0
            else:
                plume_number = [i for i,val in enumerate(list(range(self.source_pos))) if val == (x,y)][0]

        amt_decay = max(self.emit_rate[plume_number]-decay_amount,0)
        self.emit_rate[plume_number] = amt_decay

        return amt_decay

    def decay_all_plumes(self,decay_amount=0):
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
            y = np.arange(0,self.__world_size[0])
        else:
            y = np.ones(self.__world_size[1])*fix_y
            

        xx,yy = np.meshgrid(x,y)
        self.temp = np.exp(-((xx - source_pos[0])**2 + (yy - source_pos[1])**2) / tau) * (peak-trough) + trough

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