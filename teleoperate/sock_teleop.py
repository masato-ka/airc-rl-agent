from threading import Thread

import zmq


class TeleopSocket:


    def __init__(self):
        pass

    def start_process(self):
        self.process = Thread(target=self.start_server())
        self.process.daemon = True
        self.process.start()


    def start_server(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind('tcp://*:5556')
        print('================START SERVER=============')
        while True:
            message = socket.recv_string()
            print(message)
        socket.close()
        context.destroy()




