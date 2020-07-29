#!/usr/bin/python3

import csv
import socket
import uuid

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def decomment(csvfile):
    for row in csvfile:
        raw = row.split('#')[0].strip()
        if raw: yield raw

with open('List_of_Cameras.csv') as csvfile:
    data = csv.reader(decomment(csvfile), delimiter=',')
    pi_data_table = [row for row in data]
pi_dict = dict(pi_data_table)

#get ip_address and cross reference in dictionary
ipaddress = get_ip()
try:
    name = [key for (key, value) in pi_dict.items() if value == ipaddress][0]
except IndexError:
    print("network not connected, getting MAC address as name")
    name = "VtK_RPI_1"
    #name = '_'.join(['{:02x}'.format((uuid.getnode() >> ele) & 0xff) for ele in range(0,8*6,8)][::-1])