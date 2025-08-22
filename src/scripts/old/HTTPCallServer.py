import os
import queue
import struct
import time
from colorama import Fore, Style
from flask import Flask, Response, request, stream_with_context, jsonify
import librosa
import numpy as np

# Audio format constants (all files and silence must use these)
SAMPLE_RATE = 16000       # Hz
NUM_CHANNELS = 1          # mono
BITS_PER_SAMPLE = 32      
SAMPLE_WIDTH = BITS_PER_SAMPLE // 8  # in bytes
CHUNK_SIZE = 1024         # bytes per yield
S_PER_CHUNK = CHUNK_SIZE / (SAMPLE_RATE * SAMPLE_WIDTH * NUM_CHANNELS)  # duration of each chunk in ms
print("Chunk duration:", S_PER_CHUNK, "s")

VALID_DIAL_BUTTONS = list(map(str, range(10))) + ["#", "*"]

class HTTPCallServer:
    audio_queue: queue.Queue
    target_time: float
    
    def __init__(self):
        self.audio_queue = queue.Queue()
    
    def generate_wav_header(self):
        """
        Generate a WAV header for a PCM stream with a huge datasize.
        This header is sent only once at the start of the stream.
        """
        datasize = 2000 * 10**6  # use a very large data size for an "infinite" stream
        header = bytearray()
        header.extend(b"RIFF")
        header.extend(struct.pack("<I", datasize + 36))
        header.extend(b"WAVE")
        header.extend(b"fmt ")
        header.extend(struct.pack("<I", BITS_PER_SAMPLE))  # Subchunk1Size for PCM
        header.extend(struct.pack("<H", 1))   # AudioFormat PCM=1
        header.extend(struct.pack("<H", NUM_CHANNELS))
        header.extend(struct.pack("<I", SAMPLE_RATE))
        byte_rate = SAMPLE_RATE * NUM_CHANNELS * SAMPLE_WIDTH
        header.extend(struct.pack("<I", byte_rate))
        block_align = NUM_CHANNELS * SAMPLE_WIDTH
        header.extend(struct.pack("<H", block_align))
        header.extend(struct.pack("<H", BITS_PER_SAMPLE))
        header.extend(b"data")
        header.extend(struct.pack("<I", datasize))
        return bytes(header)

    def generate_audio_stream(self):
        """Generator that yields a continuous WAV stream.
        When no file is queued, it yields silence; otherwise, it streams the queued file.
        """
        # Send WAV header once.
        yield self.generate_wav_header()
        
        yield b'\x00' * CHUNK_SIZE
        self.target_time = time.time()

        # To allow quick interruption, break sleep into small increments.
        while True:
            # If a next file is queued, stream its PCM data.
            try:
                # print("waiting for data")
                data = self.audio_queue.get_nowait()
                # print("timeout passed")
                # print("data:", data)
                yield data
                self.audio_queue.task_done()
            except queue.Empty:
                # time.sleep(1)
                # print("no data")
                yield b'\x00' * CHUNK_SIZE
            finally:
                self.target_time += S_PER_CHUNK
                # print("target_time:", target_time)
                
                current_time = time.time()
                # print("current_time:", current_time)
                
                diff = self.target_time - current_time
                # print("diff:", diff)
                time.sleep(max(diff, 0))
                
    def add_file_to_queue(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            return False

        # Load audio with librosa; sr=16000 will resample if needed
        audio, _ = librosa.load(file_path, sr=16000, dtype=np.float32)
        
        # Convert the float32 audio (range -1.0 to 1.0) to PCM 16-bit integers
        # audio_pcm = (audio * 32767).astype(np.int16)
        
        # Convert the PCM data to bytes
        audio_bytes = audio.tobytes()
        
        # Chunk the data and put into the queue
        for i in range(0, len(audio_bytes), CHUNK_SIZE):
            self.audio_queue.put(audio_bytes[i:i+CHUNK_SIZE])
            
        return True
        
    def start(self):
        self.app = Flask(__name__)
        
        @self.app.route("/audio")
        def stream_audio():
            # print("session:", request)
            resp = Response(stream_with_context(self.generate_audio_stream()),
                            mimetype="audio/x-wav")
            resp.headers["Accept-Ranges"] = "none"
            return resp

        @self.app.route("/set_next", methods=["POST"])
        def set_next_file():
            """
            Set the next file to stream via a POST request.
            Expected JSON payload: {"file": "path/to/file.wav"}
            """
            data = request.json
            file_path = data.get("file")
            self.audio_queue.queue.clear()
            success = self.add_file_to_queue(file_path)
            
            if not success:
                return jsonify({"error": "File not found"}), 400
            
            return jsonify({"status": "Next file set", "file": file_path})
        
        @self.app.route("/press", methods=["POST"])
        def press():
            """
            Press a button via a POST request.
            Expected JSON payload: {"button": "button_name"}
            """
            data = request.json
            button = data.get("button")
            
            button = str(button)
            
            # assert that button is in range 0 - 9 or # or *
            if button not in VALID_DIAL_BUTTONS:
                return jsonify({"error": "Invalid button"}), 400
            
            if button == "1":
                ivr.add_file_to_queue("./data/ivr/audio/b123fa04a1704bf89bf18f4121010488.wav")
                ivr.add_file_to_queue("./data/ivr/audio/018b15d0bd9145988d1359eef94a7f7f.wav")
            elif button == "2":
                ivr.add_file_to_queue("./data/ivr/audio/95edfef484b947bb8f459527b61933ce.wav")
            else:
                ivr.add_file_to_queue("./data/ivr/audio/9ec621de85fd4dbb8e31eaee0bee23df.wav")
                ivr.add_file_to_queue("./data/ivr/audio/820284a4ef0e443395f171f66cc53606.wav")
                ivr.add_file_to_queue("./data/ivr/audio/9c17e71f5ddd4066a141414fd2802634.wav")
                
            
            print(Fore.GREEN + "Button pressed:" + button + Style.RESET_ALL)
            return jsonify({"status": "Button pressed", "button": button})
        
        self.app.run(debug=True, threaded=True)
                
if __name__ == "__main__":
    ivr = IVRService()
    ivr.add_file_to_queue("./data/ivr/audio/9ec621de85fd4dbb8e31eaee0bee23df.wav")
    ivr.add_file_to_queue("./data/ivr/audio/820284a4ef0e443395f171f66cc53606.wav")
    ivr.add_file_to_queue("./data/ivr/audio/9c17e71f5ddd4066a141414fd2802634.wav")
    ivr.start()
    