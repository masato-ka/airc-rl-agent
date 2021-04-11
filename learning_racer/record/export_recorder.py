class ExpertRecorder:

    def __init__(self, wrapper_env, vae, teleop):
        self.wrapper_env = wrapper_env
        self.vae = vae
        self.teleop = teleop

    def loop(self):
        pass

    def _image_encode(self, image):
        pass

    def _record_exprot(self, obs, action):
        pass

    def _save(self, filepath):
        pass
