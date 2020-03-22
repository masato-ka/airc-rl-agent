import threading
import time
import traitlets
import numpy as np
from jetbot import Robot

from traitlets.config.configurable import SingletonConfigurable


class MobileController(SingletonConfigurable):

#    loop = True
#    controll_thread = None
    left_v = 0.0
    right_v = 0.0
#    jitter = 1.111
    max_radius = 50.0
    speed = traitlets.Float(default_value=0.0).tag(config=True)
    radius = traitlets.Float(default_value=0.0).tag(config=True)
    
    def __init__(self, wheel_distance, robot: Robot, pass_through=False,*args, **kwargs):
        super(MobileController, self).__init__(*args, **kwargs)
        self.pass_through = pass_through
        self.wheel_distance = wheel_distance
        self.robot = robot
       
    @traitlets.observe('speed')
    def _observe_speed(self, change):
        
        self.speed = change['new'] if self.pass_through else self._input_disc(change['new'])
        self.controll()
        self.robot.set_motors(self.left_v, self.right_v)
    
    @traitlets.observe('radius')
    def _observe_radius(self, change):
        self.radius = change['new']
        self.controll()
        self.robot.set_motors(self.left_v, self.right_v)
    
    def set_control(self, speed, steering):
        self.speed = self._input_disc(speed)
        self.radius = steering
        self.controll()
        self.robot.set_motors(self.left_v, self.right_v)
    
    def _input_disc(self, input):

        negative = -1.0 if input < 0e-10 else 1.0
        input = abs(input)
        if 0.01 < input <= 0.4:
            return negative * 0.3
        elif 0.4 < input <= 0.6:
            return negative * 0.5
        elif 0.6 < input <= 0.8:
            return negative * 0.7
        elif 0.8 < input <= 1.0:
            return negative * 1.0
        else:
            return 0.0
    
    def controll(self):

        if self.radius < 0e-10:
            radius = (30 * self.radius) + (self.max_radius)
            self.left_v = ((radius-self.wheel_distance)*self.speed)/(self.max_radius-self.wheel_distance) if radius != 0.0 else self.speed
            self.right_v = ((radius+self.wheel_distance)*self.speed)/(self.max_radius+self.wheel_distance) if radius != 0.0 else self.speed
        elif self.radius > 0e-10:
            radius = (-30 * self.radius) + (self.max_radius)
            self.left_v = (radius+self.wheel_distance)*self.speed/(self.max_radius+self.wheel_distance)  if radius != 0.0 else self.speed
            self.right_v = (radius-self.wheel_distance)*self.speed/(self.max_radius-self.wheel_distance) if radius != 0.0 else self.speed
        else:
            self.right_v = self.speed
            self.left_v = self.speed
        