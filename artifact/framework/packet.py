# from scapy.all import *

PROTOCOLS = {
    6  : 'TCP',
    17 : 'UDP'
}

class Packet(object):
    """
    Packet class with all the relevant packet information
    """
    
    def __init__(self, pkt):
        """
        Initialise an empty packet and extract the fields from the pkt
        if its not None.
        """
        # initialise src and dest MACs
        self.smac = None
        self.dmac = None
        # intiialise src and dest IPs
        self.sip = None
        self.dip = None
        # initialise src and dest ports
        self.sport = None
        self.dport = None
        # initialise protocol
        self.proto = None
        self.eth_type = None
        # initialise packet length
        self.length = None
        # initialise packet time
        self.time = None

        if pkt is not None:
            # Extract the fields and set the values for the class variables
            self.extract_fields(pkt)
    
    def extract_fields(self, pkt):
        """
        Extract the relevant fields from the packet

        [Arg]
        pkt: scapy packet
        """
        # MAC Addresses
        self.smac = pkt.src
        self.dmac = pkt.dst

        self.eth_type = self.get_eth_type(pkt)

        self.sip = pkt['IP'].src if pkt.haslayer('IP') else None
        self.dip = pkt['IP'].dst if pkt.haslayer('IP') else None
        self.proto = pkt['IP'].proto if pkt.haslayer('IP') else None

        try:
            self.sport = pkt[PROTOCOLS[self.proto]].sport
            self.dport = pkt[PROTOCOLS[self.proto]].dport
        except Exception:
            self.sport = '*'
            self.dport = '*'
        
        self.time = pkt.time
        

    def get_eth_type(self, pkt):
        """
        If Ethernet layer exists, get ethernet type

        [Arg]
        pkt: scapy packet

        [Returns]
        eth_type: Ethernet type in the hex
        """
        if pkt.haslayer('Ethernet'):
            eth_type = hex(pkt['Ethernet'].type)[:2] + '0' + hex(pkt['Ethernet'].type)[2:]
        else:
            eth_type = "*"
        return eth_type
    

    def is_none(self):
        """
        Check if any of the main comparable fields of the packet is None.

        [Returns]
        True: if one of the fields is None
        False: if none of the fiels id None
        """

        if self.sip is None or self.dip is None or self.sport is None or self.dport is None:
            return True
        else:
            return False