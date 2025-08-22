from classes.Environment import Environment
from colorama import Fore
from transceivers.Transceiver import Transceiver
import numpy as np

class CallClient(Transceiver):
    def __init__(self, environment: Environment, d_type: type = np.float32):
        super().__init__("CallClient",
                         environment,
                         log_color=Fore.WHITE,
                         wait_for_input=True,
                         buffer_size=1,
                         wait_for_output=True)
        self.d_type = d_type
    
    def setup(self):
        pass
    
    def teardown(self):
        pass
    
    def process(self, data):
        data_chunk = np.frombuffer(data, dtype=self.d_type)
        return data_chunk