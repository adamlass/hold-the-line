import queue
from transceivers.Transceiver import Transceiver
from colorama import Fore
import numpy as np
import requests

AUDIO_ROUTE = "audio"
PRESS_ROUTE = "press"

class HTTPCallClient(Transceiver):
    def __init__(self, 
                chunk_size: int, 
                sample_rate: int, 
                channels: int, 
                sample_width: int, 
                d_type: type,
                url: str):
        super().__init__("AudioClient", log_color=Fore.LIGHTBLUE_EX, wait_for_input=False)
        
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.d_type = d_type
        self.block_size = self.chunk_size // (self.sample_width * self.channels)
        self.url = url
        self.audio_url = f"{self.url}/{AUDIO_ROUTE}"
    
    def setup(self):
        self.log("Connecting to stream at:", self.audio_url)
        response = requests.get(self.audio_url, stream=True)
        self.it = response.iter_content(chunk_size=self.chunk_size)
        
        # Discard the WAV header (first chunk)
        header = next(self.it)
        self.log(f"Received WAV header (length={len(header)} bytes)")
    
    def _receive(self, timeout: float | None = None) -> any:
        try:
            return next(self.it)
        except queue.Empty as e:
            print("No data received:", e)
            return None
    
    def _processing_done(self, n: int = 1):
        pass
    
    def process(self, data):
        data_chunk = np.frombuffer(data, dtype=self.d_type)
        return data_chunk
        
    def press(self, button: str) -> bool:
        """
        Send a POST request to the IVRService to simulate a button press.
        """
        assert button, "No button provided"
        body = {"button": button}
        response = requests.post(f"{self.url}/{PRESS_ROUTE}", json=body)
        return response.ok
    
    def teardown(self):
        pass