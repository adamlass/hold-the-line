

from classes.Call import Call
from classes.Dial import Action
from classes.OnlineCallStateMachine import OnlineCallStateMachine
from services.CallService import CallService
from utils import generate_unique_id
from queue import Queue



class FakeCallServiceStub(CallService):
    calls: dict
    
    def __init__(self):
        super().__init__()
        self.calls = {}
        
    def get_call(self, call_id: str) -> Call:
        fake_call_sm: OnlineCallStateMachine = self.calls.get(call_id)
        if fake_call_sm is None:
            return None
        return fake_call_sm.call
        
    def create_call(self, number: str) -> Call:
        call_id = generate_unique_id()
        call = Call(call_id, number)
        fake_call = OnlineCallStateMachine(call)
        self.calls[call_id] = fake_call
        return call

    def start_call(self, call_id: str, audio_queue: Queue) -> Call:
        fake_call_sm: OnlineCallStateMachine = self.calls.get(call_id)
        call: Call = fake_call_sm.call
        call.audio_queue = audio_queue
        fake_call_sm.run()
        return call
    
    def hangup(self, call_id: str) -> bool:
        fake_call_sm: OnlineCallStateMachine = self.calls.get(call_id)
        fake_call_sm.running = False
        return True
        
    def dial(self, call_id: str, dial: Action) -> bool:
        fake_call_sm: OnlineCallStateMachine = self.calls.get(call_id)
        if fake_call_sm is None:
            return False
        
        return fake_call_sm.dial(dial)
