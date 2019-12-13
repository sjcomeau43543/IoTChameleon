The PCAPs are the standard attack PCAPs. The TXT files are the adversarial packets + the features from the PCAP. They are the successful feature vectors that tricked the combination classifier.

from attacks folder run

python3.6 modify_pcap_combination.py -p CombinationAttackPackets/worm_packet.pcap -f ../GAN/samples/rd5/100.txt

for each of the attack packets

