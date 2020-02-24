import time
from threading import Thread
import posix_ipc
import json

class TeleopSocket:


    def __init__(self):
        self.status = False
        self.shutdown = False

    def start_process(self):
        self.process = Thread(target=self.start_server)
        self.process.daemon = True
        self.process.start()


    def start_server(self):

        mq = posix_ipc.MessageQueue("/my_q01", posix_ipc.O_CREAT )
    #
        while True:
            data = mq.receive()
            message = json.loads(data[0])

            if type(message['status']) == type(True):
                self.status = message['status']
                print(self.status)
            time.sleep(0.01)
        # context = zmq.Context()
        # socket = context.socket(zmq.REP)
        # socket.bind('tcp://*:5556')
        # print('================START SERVER=============')
        # while True:
        #     message = socket.recv_string()
        #     print(message)
        # socket.close()
        # context.destroy()




