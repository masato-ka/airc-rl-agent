import time
from threading import Thread
import json

try:
    import posix_ipc
except ImportError:
    class posix_ipc:
        pass

from .util import JUPYTER_TO_AGENT, AGENT_TO_JUPYTER


class Teleoperator:

    def __init__(self):
        self.status = False
        self.shutdown = False
        self.rx_mq = posix_ipc.MessageQueue(AGENT_TO_JUPYTER, posix_ipc.O_CREAT)
        self.tx_mq = posix_ipc.MessageQueue(JUPYTER_TO_AGENT, posix_ipc.O_CREAT)

    def start_process(self):
        self.process = Thread(target=self._polling_message)
        self.process.daemon = True
        self.process.start()

    def send_status(self, status):
        obj = {'status': status}
        self.tx_mq.send(json.dumps(obj))

    def _polling_message(self):

        while True:
            data = self.rx_mq.receive()
            message = json.loads(data[0])
            if type(message['status']) == type(True):
                self.status = message['status']
                if self.status:
                    print('STOP')
                else:
                    print('START')
            time.sleep(0.01)





