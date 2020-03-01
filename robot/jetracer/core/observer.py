try:
    from jetcam.csi_camera import CSICamera
except ImportError:
    class CSICamera: pass

class Observer:

    def __init__(self, camera_width, camera_height):
        self.camera = CSICamera(width=camera_width, height=camera_height, capture_width=camera_width,
                                capture_height=camera_height, capture_fps=60)
        self.image = None

    def start(self):
        self.camera.observe(self._callback, names='value')
        self.camera.running = True

    def stop(self):
        self.camera.running = False

    def _callback(self, change):
        img = change['new']
        # Change BGR TO RGB HWC
        self.image = img[:,:,::-1]

    def observation(self):
        while self.image is None:
            pass
        return self.image