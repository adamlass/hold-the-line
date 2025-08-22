from queue import Queue
import re
from classes.Dial import Action
from classes.Environment import Environment
from services.CallService import CallService
from transceivers.Transceiver import Transceiver
from colorama import Fore

class MotorFunction(Transceiver):
    def __init__(self, environment: Environment):
        super().__init__("MotorFunction", environment, log_color=Fore.WHITE, wait_for_input=True)
        self.command_regex = re.compile(r"(\w+)(?: (\S+))?<\|eot_id\|>") 

    def setup(self):
        pass
    
    def process(self, data: Action):
        if data is None:
            return None
        
        self.environment.apply_action(data)

    def teardown(self):
        pass
    
    def press(self, button: str = None):
        self.log("Pressing button:", button)
        
        dial: Action = Action.NONE
        
        if button:
            dial = Action(button)

        success = self.environment.apply_action(dial)
        self.log("Button press successful" if success else "Button press failed")
        
    def wait(self, args=None):
        pass
        
    def start(self, args=None):
        self.log("Starting not implemented")
        
    def ask(self, question=None):
        assert question, "No question provided"
        self.log("Asking question:", question)
        self.log("Asking not implemented")