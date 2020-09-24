import time
from logging import getLogger

from stable_baselines3.common.callbacks import BaseCallback

logger = getLogger(__name__)


class ControlCallback(BaseCallback):

    def __init__(self, teleop, verbose=0):
        super(ControlCallback, self).__init__(verbose)
        self.teleop = teleop

    def on_rollout_start(self) -> None:
        if self.teleop is not None:
            self.teleop.send_status(True)
            message = True
            while self.teleop.status:
                if message:
                    logger.info("Press START.")
                message = False
                time.sleep(0.1)

    def on_training_end(self) -> None:
        logger.info("Training is came to ending. Please shutdown software manually(Ctr+C).")

    def _on_step(self) -> bool:
        return True
