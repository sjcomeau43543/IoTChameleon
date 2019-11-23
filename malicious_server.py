'''
This is our mock server. It is /NOT/ intended to execute any malicious code, but simply prove that our attacks work.
Please run this file and then attack_packets.py.
'''
import sys
import socket
import struct

def main():
    addr = ('localhost', 12000)  # hardcoded for simplicity

    # ---- Create UDP socket ---- #
    udp_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_s.bind(addr)
    print("UDP socket successfully created; listening on port {0}".format(addr[1]))

    # ---- Receive spyware attack ---- #
    message, address = udp_s.recvfrom(1024)
    header = struct.unpack('!HHHH', message[:8])
    payload = message[8:].decode()
    print("Received spyware packet: ", header, payload)
    print("Closing UDP socket")
    udp_s.close()

    # c, addr = s.accept()
    # print("Got connection from {0}".format(addr))

    # ---- Receive attacks from client ---- #
    # while True:
    #     data = c.recv(1024)
    #     if not data:
    #         c.close()
    #         break
    #     print("Received: " + data.decode())

if __name__== "__main__":
  main()
