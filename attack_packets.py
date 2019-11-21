'''
This file describes the format and protocol of our attack packets and will run with our mock malicious server (malicious_server.py).
Please make sure malicious_server.py is running before running this file!

We simulate the following attacks: malware, spyware, worm, botnet.
The immutable fields for the packets also mentioned, any other portion of the packet can be modified without impacting the attack.
'''
import socket
import requests
import sys
from scapy.all import *

'''
(1) Malware:
HTTP/HTTPS GET request to a malicious host
- Must haves: GET request, URL (route)
- Not important: remaining packet header (e.g. Cache-Control, Connection)
'''
def malware(socket):
    url_host = 'www.google.com'
    # ---- Send HTTP request using TCP socket ---- #
    get_req = 'GET / HTTP/1.1\r\nHost: %s\r\n\r\n' % url_host
    print("Sent malware attack: " + get_req)
    socket.sendall(get_req.encode())

    # ---- Send HTTP request using request library to see full header ---- #
    r = requests.get('http://' + url_host)
    print(r.headers)

    # req = HTTP()/HTTPRequest(
    #     Accept_Encoding=b'gzip, deflate',
    #     Cache_Control=b'no-cache',
    #     Connection=b'keep-alive',
    #     Host=b'www.secdev.org',
    #     Pragma=b'no-cache'
    # )

'''
(2) Spyware:
UDP to a malicious server from: compromised IoT device & data sniffed from other devices on the network
- Must haves: destination IP and port, payload
- Not important: remainder of the packet (can add around the payload), header
'''
def spyware(socket):
    pass

'''
(3) Worm:
ICMP neighbor solicitation message with self-propagating worm code ('Hello World' code)
- Must haves: data field, type = 135
- Not important: destination, query, length
'''
def worm(socket):
    pass

'''
(4) Botnet:
Domain Name System (DNS) query to C&C [nslookup somesite.com cc_IP]
- Must haves: destination IP and port
- Not important: remainder of the query packet
'''
def botnet(socket):
    pass

def main():
    # ---- TCP Connection with server ---- #
    port_num = 12346  # hardcoded for simplicity
    host = 'localhost'

    try:
        host_ip = socket.gethostbyname(host)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host_ip, port_num))
    except socket.error as err:
        print("Socket creation failed with error {0}".format(err))
        return None
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return None

    print("Connected to malicious server at {0}".format(host_ip))

    # Send attacks to malicious server
    malware(s)
    s.close()

if __name__== "__main__":
  main()
