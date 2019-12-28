import pygame
import random
import numpy as np
import os

from time import time

# based on https://github.com/MattChanTK/gym-maze/blob/master/gym_maze/envs/maze_view_2d.py
# https://opengameart.org/content/dragons

class WorldView2D:

    def __init__(self, world_name="World2D", world_size=(512, 512)):

        # PyGame configurations
        pygame.init()
        pygame.display.set_caption(world_name)
        self.clock = pygame.time.Clock()
        self.__game_over = False

        # to show the right and bottom border
        self.screen = pygame.display.set_mode(world_size)
        self.__world_size = tuple(map(sum, zip(world_size, (-1, -1))))

        # Create the animal
        self.__worm_img = pygame.image.load('envs/res/reddragon.png')
        # https://stackoverflow.com/questions/43046376/how-to-change-an-image-size-in-pygame
        self.__worm_img = pygame.transform.scale(self.__worm_img, (10, 10))

        # Create a background
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((255, 255, 255))

        # Create a layer for the maze
        self.world_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
        self.world_layer.fill((0, 0, 255, 0,))

        # show the world
        self.view_update()


    def quit_game(self):
        try:
            self.__game_over = True
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    def view_update(self,agent=None,viewlayers=None):
        if not self.__game_over:
            t = time()
            if viewlayers is not None:
                self.__draw_world(viewlayers=viewlayers)

            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.world_layer,(0, 0))

            if agent is not None:
                self.__worm = agent
                self.__draw_worm()

            pygame.display.flip()
            # print('view_update took ' + str(time() - t))

    def __draw_world(self,viewlayers=None):
        t = time()
        # https://stackoverflow.com/questions/40755989/pygame-create-grayscale-from-2d-numpy-array
        # https://www.reddit.com/r/pygame/comments/9gyjaq/rendering_a_small_numpy_array_in_pygame/

        # line_colour = (0, 0, 0, 255)
        # ar = pygame.PixelArray(self.world_layer)
        # for x in list(range(viewlayers.shape[0])):
        #     for y in list(range(viewlayers.shape[1])):
        #         ar[x,y] = (0,np.clip(viewlayers[x,y]*255,0,255),0)

        # del ar

        # self.screen.blit(self.world_layer, (0, 0))
        # pygame.surfarray.blit_array(self.world_layer, (viewlayers*255).astype(int))
        # bv = self.world_layer.get_view('0')
        # bv.write((viewlayers*255).astype(int).tostring())
        # w = np.array([el for el in )*255])
        # print(viewlayers.shape)
        # print(np.max(viewlayers))

        # w = np.clip((np.random.random((512,512,3))*255).astype(int),0,255)
        # w2 = [tuple(list(y)) for y in w.reshape((512*512,3))]
        # w3 = np.array(w2,dtype=('int,int,int')).reshape((512,512))
        # # print(w3.shape)
        # # print(w3[0,0])
        # w = np.ones((512,512),dtype=('int,int,int,int'))
        # for x in list(range(512)):
        #     for y in list(range(512)):
        #         w[x,y] = (0,255,0,255)
        # pygame.pixelcopy.array_to_surface(self.world_layer, w)


        # # create a 3D array with 30x30x3 (the last dimension is for the RGB color)
        # cells = np.ndarray((512, 512, 3))

        # # color dictionary, represents white, red and blue
        # color_dict = {
        #         0: (255, 255, 255),
        #         1: (255, 0, 0),
        #         2: (0, 0, 255)
        #         }

        # pick a random color tuple from the color dict
        # for i in range(cells.shape[0]):
        #     for j in range(cells.shape[1]):
        #         cells[i][j] = color_dict[random.randrange(3)]

        # print(cells)
        # print(cells.shape)
        # print(cells.dtype)

        # do we need to transpose??
        cells2 = np.ndarray((viewlayers.shape[0],viewlayers.shape[1],3))
        cells2[:,:,0] = np.clip(viewlayers*255,0,255).astype(int)
        # print(cells2.shape)
        # print(cells2.dtype)
        # screen = pg.display.set_mode((WIDTH, HEIGHT))
        pygame.surfarray.blit_array(self.world_layer, cells2)


        # bv = self.world_layer.get_buffer()
        # print(bv)
        # bv.write(w.tostring(), 0)
        # pygame.pixelcopy.array_to_surface(self.world_layer,w)

        # newarr = w.view(dtype=np.dtype([('x', 'i4'), ('y', 'i4')]))
        # for x in list(range(512)):
        #     for y in list(range(512)):

        # newarr = newarr.reshape(newarr.shape[:-1])
        # pygame.surfarray.blit_array(self.world_layer, nwearr)
        # pygame.surfarray.blit_array(self.world_layer, (np.random.random((512,512,3))*255).T.astype(int))
        # ar = pygame.surfarray.pixels_green(self.world_layer)
        # for x in list(range(viewlayers.shape[0])):
        #     for y in list(range(viewlayers.shape[1])):
        #         ar[x,y] = np.clip(viewlayers[x,y]*255,0,255)
        # ar = []
        # self.screen.blit(self.world_layer, (0, 0))

        # import copy
        # ar = copy.copy((np.random.random((512,512))*255).T.astype(int))
        # print(ar)
        # ar[:255,:255] = 255
        # ar = []
        self.screen.blit(self.world_layer, (0, 0))
        # print('draw_world took ' + str(time() - t))



    def __draw_worm(self, transparency=255):
        # replace with cute pixel art

        t = time()
        self.screen.blit(self.__worm_img,(self.__worm[0],self.__worm[1]))
        # print('draw_worm took ' + str(time() - t))










