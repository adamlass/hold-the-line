import time
from classes.Call import Call
from services.FakeCallServiceStub import FakeCallServiceStub
from services.TranscriptionService import TranscriptionService
from transceivers.CallClient import CallClient
from transceivers.Printer import Printer
import numpy as np
from transceivers.Listener import Listener
from transceivers.Transcriber import Transcriber
from transceivers.LLMAgent import LLMAgent
from transceivers.MotorFunction import MotorFunction


D_TYPE = np.float32

TREE = "tree1"

call_service = FakeCallServiceStub()
call: Call = call_service.create_call(TREE)
print(call.id)

call_client = CallClient()

listener = Listener()
transcription_service = TranscriptionService()
transcriber = Transcriber(transcription_service)
transcriber2 = Transcriber(transcription_service)
printer = Printer()
llm_agent = LLMAgent()

motor = MotorFunction()

call_client.subscribe(transcriber)
call_client.subscribe(transcriber2)
call_client.subscribe(listener)
transcriber.subscribe(llm_agent)
transcriber2.subscribe(printer)
llm_agent.subscribe(motor)

printer.run()
motor.run()
llm_agent.run()
transcriber.run()
transcriber2.run()
listener.run()
call_client.run()

try:
    while True:
        time.sleep(60)
        call_client.join()
        motor.join()
        
        call: Call = call_service.create_call(TREE)
        
        call_client = CallClient(call_service, call.id)
        motor = MotorFunction(call_service, call.id)
        
        call_client.subscribe(transcriber)
        call_client.subscribe(listener)
        llm_agent.subscribe(motor)
        
        motor.run()
        call_client.run()
        
        
        
except KeyboardInterrupt:
    call_client.join()
    listener.join()
    transcriber.join()
    llm_agent.join()
    motor.join()
    # printer.join()