#!/usr/bin/python3

"""
Modified Traffic analysis engine class
"""
import yaml
import sys
import pyshark
import re
import sqlite3
import time
import hashlib


class TrafficAnalysisEngine(object):

	def __init__(self, pcap_file, device_name):
		self.filename = pcap_file
		self.devicename = 'device_fingerprint_database/' + device_name + ".yaml"
		self.device_name = device_name
		f = open(self.devicename)
		self.rules = yaml.safe_load(f)
		self.device_ip = '192.168.1.47'
		self.DNS_server_ip = []
		self.DNS_server_ip.append(self.rules['DNS Server'])
		self.domain_ip = []
		self.domain_ip.append(self.rules['domain'])
		self.new_ip = {}

	def UDP_analyze(self,packet):
		if (packet['ip'].src == self.rules['device'][0]) and (packet['ip'].dst == self.rules['packet'][1]['dst_IP']): #and (packet.info.split(' ')[0] == self.rules['packet'][1]['src_port']) and (packet.info.split(' ')[2] == self.rules['packet'][1]['dst_port']):
			return True
		else:
			return False

	# ----Added function, not in original code ---- #
	def DNS_analyze(self,packet):
		for dns_IP in self.DNS_server_ip:
			if (packet['ip'].src == self.rules['device'][0]) and (packet['ip'].dst == dns_IP): #and (packet.info.split(' ')[0] == self.rules['packet'][1]['src_port']) and (packet.info.split(' ')[2] == self.rules['packet'][1]['dst_port']):
				return True
		return False

	def HTTP_analyze(self,packet):
		for domain_IP in self.domain_ip[0]:
			if(packet['ip'].src == self.device_ip) and (packet['ip'].dst == domain_IP) :
				return True
			elif(packet['ip'].src == domain_IP) and (packet['ip'].dst == self.device_ip) :
				return True
		return False

	# ----Added function, not in original code ---- #
	def ICMP_analyze(self,packet):
		if (packet['ip'].src == self.rules['device'][0]) and (packet['ip'].dst == self.rules['packet'][4]['dst_IP']): #and (packet.info.split(' ')[0] == self.rules['packet'][1]['src_port']) and (packet.info.split(' ')[2] == self.rules['packet'][1]['dst_port']):
			return True
		else:
			return False

	def traffic_analyze(self,packet):
		prot = packet.transport_layer

		try:
			if prot == "UDP":
				if packet[3].layer_name == 'dns':
					return self.DNS_analyze(packet)
				else:
					return self.UDP_analyze(packet)

			if prot == "TCP" and packet[3].layer_name == 'http':
				return self.HTTP_analyze(packet)

			if not prot == 'None'and packet.icmp:
				return self.ICMP_analyze(packet)
		except:
			print("Protocol not accounted for")

		return True

	def run(self):
		cap = pyshark.FileCapture(self.filename)
		for p in cap:
			ret = self.traffic_analyze(p)
			if not ret:
				print("[Result] WARNING: Attack has been discovered.")
			else:
				print("[Result] No security issues.")
