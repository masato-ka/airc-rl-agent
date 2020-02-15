import numpy as np
import time
from threading import Thread

import pygame

DISPLAY_SIZE = (800,600)
KEY_MIN_DELAY = 1
class Teleoperator:

    def __init__(self):
        self.window = None
        self.status = False
        self.shutdown = False
        self.current_image = None

    def start_process(self):
        self.process = Thread(target=self.main_loop)
        self.process.daemon = True
        self.process.start()


    def set_current_image(self, image):
        self.current_image = image

    def _update_screen(self):

        if self.windonw is None:
            return

        if self.current_image is not None:
            current_image = np.swapaxes(self.current_image, 0, 1)
            if self.image_surface is None:
                self.image_surface = pygame.pixelcopy.make_surface(current_image)
            pygame.pixelcopy.array_to_surface(self.image_surface, current_image)
            self.window.blit(self.image_surface, (20, 350))

    def main_loop(self,):

        pygame.init()
        end = False

        last_time_pressed = {'space': 0, 'escape':0, 'm': 0, 't': 0, 'b': 0, 'o': 0}
        while not end:
            keys = pygame.key.get_pressed()
            self.window = pygame.display.set_mode(DISPLAY_SIZE, pygame.RESIZABLE)
            if keys[pygame.K_SPACE] and (time.time() - last_time_pressed['space']) > KEY_MIN_DELAY:
                print('Press SPACE')
                self.status = not self.status
                last_time_pressed['space'] = time.time()

            if keys[pygame.K_ESCAPE] and (time.time() - last_time_pressed['space']) > KEY_MIN_DELAY:
                end = True
                self.shutdown = True
                print('end')
                last_time_pressed['escape'] = time.time()


if __name__ == '__main__':
    tele = Teleoperator()
    tele.start_process()
    while not tele.shutdown:
       print(tele.status)
       time.sleep(1)
