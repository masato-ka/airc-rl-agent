try:
    from jetracer.nvidia_racecar import NvidiaRacecar
except ImportError:
    class NvidiaRacecar:pass

class RobotController:

    # STEERING_GAIN = -0.65 #-0.85
    # STEERING_OFFSET = 0.0
    # THROTTLE_GAIN = -0.8 #0.6
    # THROTTLE_OFFSET = 0.0

    def __init__(self, config):

        self.car = NvidiaRacecar(
            steering_channel=config.jetracer_steering_channel(),
            throttle_channel=config.jetracer_throttle_channel(),
            steering_gain=config.jetracer_steering_gain(),
            steering_offset=config.jetracer_steering_offset(),
            throttle_gain=config.jetracer_throttle_gain(),
            throttle_offset=config.jetracer_throttle_offset()
        )

    def action(self, steering, throttle):
        self.car.steering = float(steering)
        self.car.throttle = float(throttle)
