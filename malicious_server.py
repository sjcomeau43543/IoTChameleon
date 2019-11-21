'''
This is our mock server. It is /NOT/ intended to execute any malicious code, but simply prove that our attacks work.
Please run this file and then attack_packets.py.
'''
import sys
import socket

def main():
    # ---- TCP Connection with client ---- #
    port_num = 12346  # hardcoded for simplicity
    host = 'localhost'

    try:
        host_ip = socket.gethostbyname(host)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host_ip, port_num))
        s.listen(5)
        print("Socket successfully created; listening on port {0}".format(port_num))
    except socket.error as err:
        print("Socket creation failed with error {0}".format(err))
        return None
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return None

    c, addr = s.accept()
    print("Got connection from {0}".format(addr))

    # ---- Receive attacks from client ---- #
    while True:
        data = c.recv(1024)
        if not data:
            c.close()
            break
        print("Received: " + data.decode())

if __name__== "__main__":
  main()
