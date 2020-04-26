import json
import time
from threading import Thread

from .util import JUPYTER_TO_AGENT, AGENT_TO_JUPYTER

try:
    import posix_ipc
except ImportError:
    class posix_ipc:
        pass

class NotebookBackend:

    def __init__(self, callback):
        self.thread = None
        self.isStop = False
        self.rx_mq = posix_ipc.MessageQueue(JUPYTER_TO_AGENT, posix_ipc.O_CREAT)
        self.tx_mq = posix_ipc.MessageQueue(AGENT_TO_JUPYTER, posix_ipc.O_CREAT)
        self.callback = callback

    def __del__(self):
        self.isStop = True

    def start(self):
        self.thread = Thread(target=self._polling)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.isStop = True

    def send_status(self, flag):
        obj = {'status': flag}
        self.tx_mq.send(json.dumps(obj))

    def _polling(self):

        while not self.isStop:
            data = self.rx_mq.receive()
            message = json.loads(data[0])
            if type(message['status']) == type(True):
                self.status = message['status']
                self.callback(self.status)

            time.sleep(1)
