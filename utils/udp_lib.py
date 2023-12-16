import socket


class UDPServer:
    def __init__(self, addr='localhost', port=6340):
        self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.local_addr = addr
        self.port = port
        self.server_socket.bind((self.local_addr, self.port))
        print("UDP Server is running!")

    def disconnect(self):
        self.server_socket.close()
        print("Disconnection Successful")

    def receive_msg(self):
        return self.server_socket.recvfrom(1024)[1]

    def send_msg(self, message):
        self.server_socket.sendto(message, (self.local_addr, self.port))


class UDPClient:
    def __init__(self, address='localhost', port=6340):
        self.client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.addr = address
        self.port = port
        self.client_socket.connect((self.addr, self.port))
        print("UDP Client is Running!")

    def disconnect(self):
        self.client_socket.close()
        print("Disconnection Successful")

    def send_msg(self, message):
        self.client_socket.sendto(message, (self.addr, self.port))
