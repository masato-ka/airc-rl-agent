try:
    from jetbot import Camera
except ImportError:
    class Camera:pass

class Observer:

    def __init__(self, camera_width, camera_height):
        self.camera = Camera(width=camera_width, height=camera_height)
        self.image = None

    def start(self):
        self.camera.observe(self._callback, names='value')
        self.camera.start()

    def stop(self):
        self.camera.stop()

    def _callback(self, change):
        img = change['new']
        # Change BGR TO RGB HWC
        self.image = img[:,:,::-1]

    def observation(self):
        while self.image is None:
            pass
        return self.image