from classes.Environment import Environment
from transceivers.Transceiver import Transceiver
from colorama import Fore


class Printer(Transceiver):
    def __init__(self, environment: Environment):
        super().__init__("Printer", environment, log_color=Fore.MAGENTA, wait_for_input=True)

    def process(self, data):
        if data:
            self.log(data)
        return None
    
    def setup(self):
        pass
    
    def teardown(self):
        pass