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
def malware():
    packet = ''

'''
(2) Spyware:
UDP to a malicious server from: compromised IoT device & data sniffed from other devices on the network
- Must haves: destination IP and port (server_addr), payload
- Not important: remainder of the packet (can add around the payload), header

Code built off of: https://stackoverflow.com/questions/15049143/raw-socket-programming-udp-python
'''
def spyware():
    # Not modifiable
    protocol = '11' # UDP protocol
    dest_ip = '0b16212c' # pretending malicious server is at 11.22.33.44
    # dest_port = '076e' I think this can be anything as long as the server knows
    data = '4e4f5449465920414c49564520534444502f312e300d0a46726f6d3a20223139322e3136382e312e34373a31393032220d0a486f73743a2022536f6e6f732d373832384341424135423743220d0a4d61782d4167653a20313830300d0a547970653a2022736f6e6f733a5a6f6e65706c61796572220d0a5072696d6172792d50726f78793a20226d656469615f73657276696365220d0a50726f786965733a20226d656469615f736572766963652c616d706c6966696572220d0a4d616e7566616374757265723a2022536f6e6f73220d0a4d6f64656c3a20225a6f6e65706c61796572220d0a4472697665723a2022736f6e6f732e63347a220d0a'  # data from sniffed packet

    # Modifiable
    ethernet_layer = '01005e7ffffa7828caba5b7c0800'
    ip_layer = '45000118cefb400020' + protocol + 'd907c0a8012f' + dest_ip
    udp_layer = 'aae7076e01042d9a'
    packet = ethernet_layer + ip_layer + udp_layer + data

    return packet

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
    # ---- Spyware attack ---- #
    print("Spyware packet:\n" + spyware())

    # ---- Malware attack ---- #
    #print("Malware packet:\n" + malware())

if __name__== "__main__":
  main()