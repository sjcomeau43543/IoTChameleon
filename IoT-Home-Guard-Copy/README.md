Modified from: https://github.com/arthastang/IoT-Home-Guard

To run:
  - $ python3 software_tools/setup.py install
  - It may be necessary to manually copy the correct IoT-Home-Guard.py from the software_tools/build.scripts-{python_version}/IoT-Home-Guard.py to the software_tools directory
  - Create a yaml file in software_tools/device_fingerprint_database for your device
  - Set the pcap_file to be a path to your device's pcap file in software_tools/IoT-Home-Guard.py
  - Update software_tools/traffic_analysis_engine/traffic_analysis_engine.py
    - Create functions to analyze the protocols listed in the yaml file --> be sure to list the proper index in when using self.rules['packet'][#]
    - Call the functions in traffic_analyze(self,packet)
  - $ python3 software_tools/IoT-Home-Guard.py

To create pcaps from text file:
  - Make sure text file (packet.txt) is in the form of a hex dump (rows are labelled and there are spaces between bytes)
  - $ text2pcap -d packet.txt packet.pcap
