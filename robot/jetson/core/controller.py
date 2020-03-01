from jetracer.nvidia_racecar import NvidiaRacecar


class RobotController:

    def __init__(self):
        car = NvidiaRacecar()


    def action(self, steering, throttle):
        self.car.steering = float(steering)
        self.car.throttle = float(throttle)