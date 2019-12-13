import os
import sys
import argparse

# COMBO
sys.path.insert(1, "../COMBO")

from device_sequence_classifier import DeviceSequenceClassifier
import utils
import pandas as pd
import numpy as np

# packet mod
from pcapfile import savefile
from pcapfile.protocols.linklayer import ethernet
from pcapfile.protocols.network import ip
from pcapfile.protocols.transport import tcp
from pcapfile.protocols.transport import udp
import binascii


def get_data(dataset_csv, use_cols, device, dsc):
    validation = utils.load_data_from_csv(dataset_csv, use_cols=use_cols)

    all_sess = dsc.split_data(validation)[0]
    other_dev_sess = validation.groupby(dsc.y_col).get_group(device)
    other_dev_sess = dsc.split_data(other_dev_sess)[0]

    classification = 1 if device == device else 0

    # get the optimal sequence length for data
    opt_seq_len = dsc.find_opt_seq_len(validation)
    seqs = []
    for i in range(opt_seq_len):
        seqs.append(other_dev_sess[i])

    # return sequences
    return seqs

# start argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pcap', help='the pcap to modify as an attack packet that passes the IDS', required=True)
parser.add_argument('-f', '--features', help='the feature vector from the Combination project or the GAN', required=True)

args = parser.parse_args()


# classifiers
classifnames = { 'baby_monitor'   : 'baby_monitor_cart_gini_200_samples_leaf',
                'lights'         : 'lights_cart_gini_200_samples_leaf',
                'motion_sensor'  : 'motion_sensor_cart_entropy_200_samples_leaf',
                'security_camera': 'security_camera_cart_entropy_200_samples_leaf',
                'smoke_detector' : 'smoke_detector_cart_entropy_200_samples_leaf',
                'socket'         : 'socket_cart_entropy_200_samples_leaf',
                # 'thermostat'     : '',
                # 'tv'             : 'tv_cart_gini_50_samples_leaf',
                'watch'          : 'watch_cart_entropy_100_samples_leaf',
                # 'water_sensor'   : 'water_sensor_cart_entropy_100_samples_leaf'
              }

classifiers = { #'baby_monitor'   : DeviceSequenceClassifier('../COMBO/models', "../COMBO/models/baby_monitor/"+classifnames['baby_monitor']+".pkl", use_cols="../COMBO/data/use_cols.csv", is_model_pkl=True),
                #'lights'         : DeviceSequenceClassifier('../COMBO/models', "../COMBO/models/lights/"+classifnames['lights']+".pkl", use_cols="../COMBO/data/use_cols.csv", is_model_pkl=True),
                #'motion_sensor'  : DeviceSequenceClassifier('../COMBO/models', "../COMBO/models/motion_sensor/"+classifnames['motion_sensor']+".pkl", use_cols="../COMBO/data/use_cols.csv", is_model_pkl=True),
                #'security_camera': DeviceSequenceClassifier('../COMBO/models', "../COMBO/models/security_camera/"+classifnames['security_camera']+".pkl", use_cols="../COMBO/data/use_cols.csv", is_model_pkl=True),
                #'smoke_detector' : DeviceSequenceClassifier('../COMBO/models', "../COMBO/models/smoke_detector/"+classifnames['smoke_detector']+".pkl", use_cols="../COMBO/data/use_cols.csv", is_model_pkl=True),
                #'socket'         : DeviceSequenceClassifier('../COMBO/models', "../COMBO/models/socket/"+classifnames['socket']+".pkl", use_cols="../COMBO/data/use_cols.csv", is_model_pkl=True),
                'watch'          : DeviceSequenceClassifier('../COMBO/models', "../COMBO/models/watch/"+classifnames['watch']+".pkl", use_cols="../COMBO/data/use_cols.csv", is_model_pkl=True),
              }

data        = { #'baby_monitor'   : get_data('../COMBO/data/validation.csv', pd.read_csv(os.path.abspath("../COMBO/data/use_cols.csv")), 'baby_monitor', classifiers['baby_monitor']),
                #'lights'         : get_data('../COMBO/data/validation.csv', pd.read_csv(os.path.abspath("../COMBO/data/use_cols.csv")), 'lights', classifiers['lights']),
                #'motion_sensor'  : get_data('../COMBO/data/validation.csv', pd.read_csv(os.path.abspath("../COMBO/data/use_cols.csv")), 'motion_sensor', classifiers['motion_sensor']),
                #'security_camera': get_data('../COMBO/data/validation.csv', pd.read_csv(os.path.abspath("../COMBO/data/use_cols.csv")), 'security_camera', classifiers['security_camera']),
                #'smoke_detector' : get_data('../COMBO/data/validation.csv', pd.read_csv(os.path.abspath("../COMBO/data/use_cols.csv")), 'smoke_detector', classifiers['smoke_detector']),
                #'socket'         : get_data('../COMBO/data/validation.csv', pd.read_csv(os.path.abspath("../COMBO/data/use_cols.csv")), 'socket', classifiers['socket']),
                'watch'          : get_data('../COMBO/data/validation.csv', pd.read_csv(os.path.abspath("../COMBO/data/use_cols.csv")), 'watch', classifiers['watch']),
              }


use_columns = pd.read_csv(os.path.abspath("../COMBO/data/use_cols.csv"))

# open the attack packet
pc = open(args.pcap, 'rb')
attack = savefile.load_savefile(pc, verbose=True)
raw = attack.packets[0].raw()

eth_frame = ethernet.Ethernet(raw)

# only for spyware packet 

'''ip_packet = ip.IP(binascii.unhexlify(eth_frame.payload))

# fields I need to add to advers features from packet
ack = ip_packet.flags ^ 2
bytes = ip_packet.len - 20
http_GET = 1
http_POST = 0
packets = 1
ttl = ip_packet.ttl
is_http = 1
is_port_80 = 1
suffix_is_com = 1'''

# for all other packets
tcp_packet = tcp.TCP(binascii.unhexlify(eth_frame.payload))
# fields I need to add to advers features from packet
ack = tcp_packet.ack
bytes = tcp_packet.sum
http_GET = 1
http_POST = 0
packets = 1
is_http = 1
is_port_80 = 1
suffix_is_com = 1

# open the adversarial features
with open(args.features) as f:
    feat = [int(x) for x in next(f).split()]
print(feat)
uc = use_columns.columns.to_list()
uc.remove("device_category")
attack_adv_features = pd.DataFrame([feat], columns=uc)

attack_adv_features['ack'] = ack
attack_adv_features['bytes'] = bytes
attack_adv_features['http_GET'] = http_GET
attack_adv_features['http_POST'] = http_POST
attack_adv_features['packets'] = packets
# attack_adv_features['ttl_A_avg'] = ttl
attack_adv_features['is_http'] = is_http
attack_adv_features['B_port_is_80'] = is_port_80
attack_adv_features['suffix_is_com'] = suffix_is_com


# attempt attack against watch classifier
if classifiers['watch'].model.predict(attack_adv_features):
    print("Attack succeeded")
    fname = str(args.pcap).strip(".pcap")+".txt"
    np.savetxt(fname, attack_adv_features, fmt='%d')
    print("Wrote", fname)
else:
    print("Attack failed, continue to update parameters or use a different adversarial packet")

