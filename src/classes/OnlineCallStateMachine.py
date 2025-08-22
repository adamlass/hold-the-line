
import json
import os
from queue import Empty, Queue
from threading import Lock, Thread
import time
from classes import Segment, Content
from classes.Call import Call
from classes.ConditionalBlocker import ConditionalBlocker
from classes.Dial import Action
from colorama import Fore, Style
import librosa
import numpy as np

DATA_PATH = "data/trees"

SAMPLE_RATE = 16000       # Hz
NUM_CHANNELS = 1          # mono
BITS_PER_SAMPLE = 32      
SAMPLE_WIDTH = BITS_PER_SAMPLE // 8  # in bytes
CHUNK_SIZE = 1024         # bytes per yield
S_PER_CHUNK = CHUNK_SIZE / (SAMPLE_RATE * SAMPLE_WIDTH * NUM_CHANNELS)  # duration of each chunk in ms
print("Chunk duration:", S_PER_CHUNK, "s")
BUFFER_SECONDS = 0.5

class OnlineCallStateMachine:
    call: Call
    call_tree: Segment
    current_segment: Segment
    running: bool
    audio_buffer: Queue
    content_queue: Queue
    next_segment: Segment
    blocker: ConditionalBlocker
    segment_lock: Lock = Lock()
    
    def __init__(self, call: Call, data_path: str = DATA_PATH):
        self.call = call
        self.log_color = Fore.CYAN
        self.name = "OnlineCallStateMachine"
        self.current_segment = None
        self.running = False
        self.audio_buffer = Queue()
        self.next_segment = None
        self.blocker = ConditionalBlocker(blocking=True, logger=self.log)
        
        # load json file
        self.file_path = f"{data_path}/{call.number}/callTreeReplaced.json"
    
    def run(self) -> Thread:
        self.setup()
        self.running = True
        self.thread = Thread(target=self.loop)
        self.thread.start()
        self.log("Started")
        return self.thread
        
    def setup(self):
        self.call_tree: Segment = json.load(open(self.file_path))
        self.current_segment = self.call_tree
        self.run_segment(self.current_segment)
    
    def log(self, *args, flush: bool=True, end: str="\n"):
        print(f"{self.log_color}{self.name}:", *args, Style.RESET_ALL, flush=flush, end=end)
    
    def join(self):
        self.running = False
        self.thread.join()
        self.log("Stopped")
    
    def loop(self):
        data = b'\x00' * CHUNK_SIZE
        self.call.audio_queue.put_nowait(data)
        
        self.start_time = time.time()
        self.content_duration = 0
        self.lost_time = 0
        self.content_pointer = lambda: self.start_time + self.content_duration + self.lost_time
        
        while self.running:
            try:
                with self.blocker:
                    data = self.audio_buffer.get_nowait()
                
                self.audio_buffer.task_done()
                silence = False
                
            except Empty:
                data = b'\x00' * CHUNK_SIZE
                silence = True
                if self.next_segment:
                    self.run_segment(self.next_segment, clear_buffer=True)
            finally:
                current_time = time.time()
                lost_time = max(current_time - self.content_pointer(), 0)
                
                self.lost_time += lost_time
                   
                self.call.audio_queue.put_nowait(data)
                self.content_duration += S_PER_CHUNK
                
                target_time = current_time + BUFFER_SECONDS
                diff = self.content_pointer() - target_time
                # self.log(f"Duration - Actual: {(current_time - self.start_time):.3f},  Target: {(target_time - self.start_time):.3f}, Content:{(self.content_pointer() - self.start_time):.3f}, Lost: {self.lost_time:.3f}. diff: {diff:.3f}, silence: {silence}")
                if diff < S_PER_CHUNK:
                    time.sleep(max(diff, 0))
                    
        current_time = time.time()
        total_duration = current_time - self.start_time
        self.log(f"Total duration: {total_duration:.3f}, Total lost time: {self.lost_time:.3f} ({(self.lost_time / total_duration * 100):.2f}%)")
        self.log("Exiting loop")
    
    def add_file_to_queue(self, file_path: str) -> bool:
        self.log("Adding file to queue:", file_path)
        if not os.path.exists(file_path):
            return False

        # Load audio with librosa; sr=16000 will resample if needed
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, dtype=np.float32)
        
        # Convert the float32 audio (range -1.0 to 1.0) to PCM 16-bit integers
        # audio_pcm = (audio * 32767).astype(np.int16)
        
        # Convert the PCM data to bytes
        audio_bytes = audio.tobytes()
        
        # Chunk the data and put into the queue
        for i in range(0, len(audio_bytes), CHUNK_SIZE):
            self.audio_buffer.put(audio_bytes[i:i+CHUNK_SIZE])
            
        return True

    def clear_buffer(self):
        self.log("Clearing audio segment queue")
        with self.audio_buffer.mutex:
            self.audio_buffer.queue.clear()
    
    def run_segment(self, segment: Segment, clear_buffer: bool = False):
        with self.segment_lock:
            self.next_segment = None
            if clear_buffer:
                self.clear_buffer()
            self.current_segment = segment
            segment_type = segment["type"]
            self.log("Running segment type:", segment_type)
            
            for content in segment["contents"]:
                file_id = content["fileID"]
                file_path = f"{DATA_PATH}/{self.call.number}/speech/{file_id}.wav"
                self.add_file_to_queue(file_path)
            
            if segment_type == "welcome" and segment["nextSegment"]:
                self.next_segment = segment["nextSegment"]
    
    def dial(self, dial: Action):
        if self.current_segment["type"] == "dialOptions":
            self.log("Dialing:", str(dial))
            
            # find the segment with the matching dial option
            for content in self.current_segment["contents"]:
                index: str = str(content["index"])
                if index == str(dial.value):
                    next_segment = content["nextSegment"]
                    self.run_segment(next_segment, clear_buffer=True)
                    return True
            
            self.log("Invalid dial option")
            invalid_id = self.current_segment["invalidOption"]["fileID"]
            invalid_file_path = f"{DATA_PATH}/{self.call.number}/speech/{invalid_id}.wav"
            
            self.clear_buffer()
            self.add_file_to_queue(invalid_file_path)
            self.run_segment(self.current_segment, clear_buffer=False)

        else:
            self.log("Dialing not allowed in current segment")
            
        return False
    
if __name__ == "__main__":
    from classes.Call import Call
    queue = Queue()
    call = Call("1234", "tree1", queue)
    fsm = OnlineCallStateMachine(call)
    fsm.run()
    
    try:
        while True:
            time.sleep(1000)
    except KeyboardInterrupt:
        fsm.join()
    