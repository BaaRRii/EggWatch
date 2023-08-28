import socket
class EthernetSender:
    def __init__(self, ip, port):
        self.ipDest = ip
        self.portDest = port
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except Exception as e:
            print(e)
            raise Exception("Problem with socket")
        
    def sendMessage(self, msg):
        self.sock.sendto(msg.encode(), (self.ipDest, self.portDest))
        
    def close(self):
        if self.sock:
            self.sock.close()