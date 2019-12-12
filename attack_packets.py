'''
This file describes the format and protocol of our attack packets and will run with our mock malicious server (malicious_server.py).
Please make sure malicious_server.py is running before running this file!

We simulate the following attacks: malware, spyware, worm, botnet.
The immutable fields for the packets also mentioned, any other portion of the packet can be modified without impacting the attack.
'''
import struct
import requests
import sys
import socket
import time
from scapy.all import *
from requests_toolbelt.utils import dump

'''
(1) Malware:
HTTP/HTTPS GET request to a malicious host
- Must haves: GET request, URL (route)
- Not important: remaining packet header (e.g. Cache-Control, Connection)
'''
def malware(sock):
    url_host = 'www.google.com'
    # ---- Send HTTP request using TCP socket ---- #
    get_req = 'GET / HTTP/1.1\r\nHost: %s\r\n\r\n' % url_host
    print("Sent malware attack: " + get_req)
    sock.sendall(get_req.encode())

    # ---- Send HTTP request using request library to see full packet ---- #
    response = requests.get('http://' + url_host)
    data = dump.dump_response(response, response_prefix=b'RESPONSE: ')
    request = data[:data.find(b'RESPONSE: ')]
    #data = dump._dump_request_data(response.request, dump.PrefixSettings(b'< ', b'> '), bytearray())
    print(request.decode())

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
- Must haves: destination IP and port (server_addr), payload
- Not important: remainder of the packet (can add around the payload), header

Code built off of: https://stackoverflow.com/questions/15049143/raw-socket-programming-udp-python
'''
def spyware(sock, server_addr):
    # Not modifiable
    payload = 'spyware'  # data from sniffed packet
    dest_port = server_addr[1]

    # Modifiable
    src_prt = 12346
    length = 8 + len(payload)
    checksum = 0

    udp_header = struct.pack('!HHHH', src_prt, dest_port, length, checksum)
    packet = udp_header + payload.encode()
    print("Sent spyware attack: ", packet)
    sock.sendto(udp_header + payload.encode(), server_addr)
    return

'''
(3) Worm:
ICMP neighbor solicitation message with self-propagating worm code ('Hello World' code)
- Must haves: data field, type = 135
- Not important: destination, query, length
'''
def worm(sock):
    pass

'''
(4) Botnet:
Domain Name System (DNS) query to C&C [nslookup somesite.com cc_IP]
- Must haves: destination IP and port
- Not important: remainder of the query packet
'''
def botnet(sock):
    pass

def main():
    server_addr = ('localhost', 12000)  # hardcoded for simplicity

    # ---- UDP Connection with server ---- #
    udp_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Created UDP socket")

    # ---- Spyware attack ---- #
    spyware(udp_s, server_addr)

    print("Closing UDP socket")
    udp_s.close()
    time.sleep(1)

    # ---- TCP Connection with server ---- #
    tcp_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_s.connect(server_addr)

    print("Connected to TCP socket at {0}".format(server_addr))

    # ---- Malware attack ---- #
    malware(tcp_s)
    print("Closing TCP socket")
    tcp_s.close()

if __name__== "__main__":
  main()
