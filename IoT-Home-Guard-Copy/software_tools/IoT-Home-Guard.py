# from data_flow_catcher.data_flow_catcher import *
from traffic_analysis_engine.traffic_analysis_engine import *
import time

if __name__ == "__main__":
    #current_time = str(int(time.time()))
    #catcher = DataFlowCatcher(current_time)
    #catcher.run()

    # device_name = raw_input("Input a device name:")

    # engine = TrafficAnalysisEngine(current_time, device_name)
    # engine.run()

    #pcap_file = '/Users/edamame/Downloads/sonos_speaker.pcap'
    pcap_file = '/Users/edamame/Downloads/CS558_comp_net/IoTChameleon/Attacks/IoTHomeGuardAttacks/spyware_packet.pcap'
    engine = TrafficAnalysisEngine(pcap_file, "sonos")
    engine.run()
