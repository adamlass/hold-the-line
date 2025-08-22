import time
from classes.Call import Call
from scripts.whisper_online import ASRBase
from services.TranscriptionService import TranscriptionService
from services.FakeCallServiceStub import FakeCallServiceStub
from transceivers.CallClient import CallClient
from transceivers.Printer import Printer
import numpy as np
# from transceivers.Listener import Listener
from transceivers.Transcriber import Transcriber
# from transceivers.LLMService import LLMService
# from transceivers.MotorFunction import MotorFunction

CHUNK_SIZE = 1024         # bytes per chunk (as in IVRService)
SAMPLE_RATE = 16000       # Hz
CHANNELS = 1              # mono
SAMPLE_WIDTH = 4          # bytes per sample
D_TYPE = np.float32

TREE = "tree1"

call_service = FakeCallServiceStub()
call: Call = call_service.create_call(TREE)
print(call.id)

call_client = CallClient(call_service, call.id, D_TYPE)

# listener = Listener(SAMPLE_RATE,
#                        CHANNELS,
#                        D_TYPE,
#                        SAMPLE_WIDTH,
#                        CHUNK_SIZE)
asr_service = TranscriptionService()
transcriber = Transcriber(asr_service.asr_backend)
printer = Printer()
# llm = LLMService()

# motor = MotorFunction(call_service, call.id)

call_client.subscribe(transcriber)
# call_client.subscribe(listener)
transcriber.subscribe(printer)
# asr.subscribe(llm)
# llm.subscribe(motor)

printer.run()
# motor.run()
# llm.run()
transcriber.run()
# listener.run()
call_client.run()

try:
    while True:
        time.sleep(60)
        # call_client.join()
        # # motor.join()
        
        # call: Call = call_service.create_call(TREE)
        
        # call_client = CallClient(call_service, call.id)
        # # motor = MotorFunction(call_service, call.id)
        
        # call_client.subscribe(asr)
        # # call_client.subscribe(listener)
        # # llm.subscribe(motor)
        
        
        # # motor.run()
        # call_client.run()
        
        
        
except KeyboardInterrupt:
    call_client.join()
    # listener.join()
    transcriber.join()
    # llm.join()
    # motor.join()
    printer.join()