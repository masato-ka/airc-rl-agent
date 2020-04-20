try:
    from jetracer.nvidia_racecar import NvidiaRacecar
except ImportError:
    class NvidiaRacecar:pass

class RobotController:

    STEERING_GAIN = -0.65
    STEERING_OFFSET = 0.0
    THROTTLE_GAIN = -0.8
    THROTTLE_OFFSET = 0.0

    def __init__(self):
        self.car = NvidiaRacecar(
            steering_gain=self.STEERING_GAIN,
            steering_offset=self.STEERING_OFFSET,
            throttle_gain=self.THROTTLE_GAIN,
            throttle_offset=self.THROTTLE_OFFSET
        )

    def action(self, steering, throttle):
        self.car.steering = float(steering)
        self.car.throttle = float(throttle)
