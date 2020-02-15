import numpy as np
import time
from threading import Thread

import pygame

DISPLAY_SIZE = (800,600)
KEY_MIN_DELAY = 1

GREY = (187, 179, 179)

pygame.font.init()
SMALL_FONT = pygame.font.SysFont('Open Sans', 20)


class Teleoperator:

    def __init__(self):
        self.window = None
        self.status = False
        self.shutdown = False
        self.current_image = None
        self.reconst_image = None
        self.image_surface = None
        self.decoded_surface = None

    def start_process(self):
        self.process = Thread(target=self.main_loop)
        self.process.daemon = True
        self.process.start()


    def set_current_image(self, image):
        # RGB HWC
        self.current_image = image

    def set_reconst_image(self, image):
        # RGB CWH
        self.reconst_image = image

    def clear(self):
        self.window.fill((0, 0, 0))

    def write_text(self, text, x, y, font, color=GREY):
        text = str(text)
        text = font.render(text, True, color)
        self.window.blit(text, (x, y))

    def _update_screen(self):

        if self.window is None:
            return
        self.clear()
        help_str = 'Use arrow keys to move, q or ESCAPE to exit.'
        self.write_text(help_str, 20, 50, SMALL_FONT)

        if self.current_image is not None:
            current_image = np.swapaxes(self.current_image, 0, 1)
            print(current_image.shape)
            if self.image_surface is None:
                self.image_surface = pygame.pixelcopy.make_surface(current_image)
            pygame.pixelcopy.array_to_surface(self.image_surface, current_image)
            self.window.blit(self.image_surface, (20, 350))

        if self.reconst_image is not None:
            reconst_image = np.swapaxes(self.reconst_image, 0, 1)
            print(reconst_image.shape)
            if self.decoded_surface is None:
                self.decoded_surface = pygame.pixelcopy.make_surface(reconst_image)
            pygame.pixelcopy.array_to_surface(self.decoded_surface, reconst_image)
            self.window.blit(self.decoded_surface, (220, 350))

    def main_loop(self,):

        pygame.init()
        self.window = pygame.display.set_mode(DISPLAY_SIZE, pygame.RESIZABLE)

        end = False

        last_time_pressed = {'space': 0, 'escape': 0, 'm': 0, 't': 0, 'b': 0, 'o': 0}
        while not end:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            time.sleep(0.1)
            if keys[pygame.K_SPACE] and (time.time() - last_time_pressed['space']) > KEY_MIN_DELAY:
                print('Press SPACE')
                self.status = not self.status
                last_time_pressed['space'] = time.time()

            if keys[pygame.K_ESCAPE] and (time.time() - last_time_pressed['space']) > KEY_MIN_DELAY:
                end = True
                self.shutdown = True
                print('end')
                last_time_pressed['escape'] = time.time()
            self._update_screen()
            pygame.display.update()

if __name__ == '__main__':
    tele = Teleoperator()
    tele.start_process()
    while not tele.shutdown:
       print(tele.status)
       time.sleep(1)
