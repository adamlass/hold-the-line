from classes.Environment import Environment
import numpy as np
from transceivers.Transceiver import Transceiver
from colorama import Fore
from utils import get_best_device

CHUNK_SIZE = 1024                                       # bytes per chunk (as in IVRService)
SAMPLE_RATE = 16000                                     # Hz
CHANNELS = 1                                            # mono
SAMPLE_WIDTH = 4                                        # bytes per sample
BLOCK_SIZE = CHUNK_SIZE // (SAMPLE_WIDTH * CHANNELS)    # number of samples per chunk
D_TYPE = np.float32                                    # data type of samples

class Listener(Transceiver):
    def __init__(self, environment: Environment):
        super().__init__("Listener", environment, log_color=Fore.BLUE, wait_for_input=False)
    
    def audio_callback(self, outdata, frames, time_info, status):
        data = self._receive()
        if data is None:
            outdata[:, 0] = 0
        else:
            if len(data) < frames:
                outdata[:len(data), 0] = data
                outdata[len(data):, 0] = 0
            else:
                outdata[:, 0] = data[:frames]
            self._processing_done()
    
    def setup(self):
        if get_best_device() == "cuda":
            for transceiver_subscribed_to in self.subscribed_to:
                self.log("Unsubscribing from transceiver:", transceiver_subscribed_to.name)
                transceiver_subscribed_to.unsubscribe(self)
            
            self.log("Receiving any remaining data")
            self._processing_done(n = self.input_queue.qsize())
            
            self.log("Deleting input queue")
            self.input_queue = None
            
            self.log("CUDA device detected, skipping setup steps.")
            return
        
        import sounddevice as sd
        self.log("Setting up")
        self.stream = sd.OutputStream(channels=CHANNELS,
                            samplerate=SAMPLE_RATE,
                            dtype=D_TYPE,
                            callback=self.audio_callback,
                            blocksize=BLOCK_SIZE)
        self.stream.start()
        self.log("Audio playback started.")
    
    def _loop(self, environment: Environment):
        self.environment = environment
        if get_best_device() == "cuda":
            self.log("CUDA device detected, stopping listener.")
            self.running = False
            return
        
        import sounddevice as sd
        try:
            while self.running:
                sd.sleep(10)
        except KeyboardInterrupt:
            self.log("Playback stopped by user.")
        self.log("Exiting loop")
        
    def process(self, data):
        # Since we're handling playback via the audio_callback, there's no need to process data here.
        pass
    
    def teardown(self):
        self.log("Tearing down")
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        self.log("Audio playback stopped.")