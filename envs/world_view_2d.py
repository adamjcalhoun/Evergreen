import pygame
import random
import numpy as np
import os

from time import time

# based on https://github.com/MattChanTK/gym-maze/blob/master/gym_maze/envs/maze_view_2d.py
# https://opengameart.org/content/dragons
# http://pymedia.org/tut/src/make_video.py.html

class WorldView2D:

    def __init__(self, world_name="World2D", world_size=(512, 512), world_view_size=None):

        # PyGame configurations
        pygame.init()
        pygame.display.set_caption(world_name)
        self.clock = pygame.time.Clock()
        self.__game_over = False

        # to show the right and bottom border
        self.screen = pygame.display.set_mode(world_view_size)
        # self.__world_size = tuple(map(sum, zip(world_size, (-1, -1))))
        self.__world_size = world_size
        # print('world view: ' + str(self.__world_size))
        if world_view_size is None:
            self.__world_view_size = world_size
        else:
            self.__world_view_size = world_view_size
        self.__view_scale = (self.__world_view_size[0] / world_size[0],self.__world_view_size[1] / world_size[1])

        # Create the animal
        self.__worm_img = pygame.image.load('envs/res/creature.png')
        img_sz = self.__worm_img.get_size()
        # https://stackoverflow.com/questions/43046376/how-to-change-an-image-size-in-pygame
        self.__worm_img = pygame.transform.scale(self.__worm_img, (int((img_sz[0]/500)*2*self.__view_scale[0]), int((img_sz[1]/500)*2*self.__view_scale[1])))
        self.__worm = np.zeros(2)

        # Create a background
        self.background = pygame.Surface(world_size).convert()
        self.background.fill((255, 255, 255))
        self.__background_surf = pygame.Surface(self.__world_view_size).convert()

        # Create a layer for the maze
        self.world_layer = pygame.Surface(world_size).convert_alpha()
        self.world_layer.fill((0, 0, 255, 0,))
        self.__world_layer_surf = pygame.Surface(self.__world_view_size).convert()

        # show the world
        self.view_update()


    def quit_game(self):
        try:
            self.__game_over = True
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    def view_update(self,agent=None,viewlayers=None,agent_angle=0):
        if not self.__game_over:
            t = time()
            if viewlayers is not None:
                self.__draw_world(viewlayers=viewlayers)

            self.__background_surf = pygame.transform.scale(self.background,self.__world_view_size)
            self.__world_layer_surf = pygame.transform.scale(self.world_layer,self.__world_view_size)
            self.screen.blit(self.__background_surf, (0, 0))
            self.screen.blit(self.__world_layer_surf,(0, 0))

            if agent is not None:
                # self.__worm = agent
                self.__worm[0] = agent[0]
                self.__worm[1] = agent[1]
                self.__worm[0] *= self.__view_scale[0]
                self.__worm[1] *= self.__view_scale[1]
                # print('scaled! scaled!')
                self.__draw_worm(angle=agent_angle)

            pygame.display.flip()
            # print('view_update took ' + str(time() - t))

    def __draw_world(self,viewlayers=None):
        # t = time()

        cells2 = np.ndarray((viewlayers[0].shape[0],viewlayers[0].shape[1],3))
        # print(len(viewlayers))
        for ii,layer in enumerate(viewlayers):
            # cells2[:,:,ii] = np.clip(layer*255,0,255).astype(int)
            cells2[:,:,int(ii)] = np.clip((layer - np.mean(layer))/np.std(layer)*64 + 128,0,255).astype(int)
        pygame.surfarray.blit_array(self.world_layer, cells2)

        self.screen.blit(self.world_layer, (0, 0))



    def __draw_worm(self, transparency=255, angle=0):
        if angle != 0:
            rotated_image = pygame.transform.rotate(self.__worm_img,angle)
        else:
            rotated_image = self.__worm_img
        self.screen.blit(rotated_image,(self.__worm[0],self.__worm[1]))










