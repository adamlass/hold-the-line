import os
import time
import struct
from flask import Flask, Response, request, stream_with_context, jsonify

app = Flask(__name__)

# Audio format constants (all files and silence must use these)
SAMPLE_RATE = 24000       # Hz
NUM_CHANNELS = 1          # mono
BITS_PER_SAMPLE = 16      # 16-bit PCM
SAMPLE_WIDTH = BITS_PER_SAMPLE // 8  # in bytes
CHUNK_SIZE = 1024         # bytes per yield

# Global variable to determine the next file to stream.
# When None, we stream silence.
next_file = None

def generate_wav_header():
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
    header.extend(struct.pack("<I", 16))  # Subchunk1Size for PCM
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

def generate_audio_stream():
    """Generator that yields a continuous WAV stream.
       When no file is queued, it yields silence; otherwise, it streams the queued file.
    """
    # Send WAV header once.
    yield generate_wav_header()
    global next_file

    # To allow quick interruption, break sleep into small increments.
    sleep_counter = 0
    while True:
        # If a next file is queued, stream its PCM data.
        if next_file is not None and os.path.exists(next_file):
            sleep_counter = 0
            with open(next_file, "rb") as f:
                print("sending file")
                # Skip the 44-byte header
                f.seek(44)
                while True:
                    data = f.read(CHUNK_SIZE)
                    if not data:
                        break
                    yield data
                    # time.sleep(chunk_duration)
            # After streaming the file, reset next_file.
            next_file = None
        else:
            # print("silence")
            if sleep_counter % 10 == 0:
                print("sending silence", sleep_counter)
                yield b'\x01' * CHUNK_SIZE
                # Sleep for a long time if no file is queued.
            sleep_counter += 1
            time.sleep(0.1)
            

@app.route("/audio")
def stream_audio():
    global next_file
    next_file = "./data/ivr/audio/test0.wav"
    resp = Response(stream_with_context(generate_audio_stream()),
                    mimetype="audio/x-wav")
    resp.headers["Accept-Ranges"] = "none"
    return resp

@app.route("/set_next", methods=["POST"])
def set_next_file():
    """
    Set the next file to stream via a POST request.
    Expected JSON payload: {"file": "path/to/file.wav"}
    """
    global next_file
    data = request.json
    file_path = data.get("file")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 400
    next_file = file_path
    return jsonify({"status": "Next file set", "file": file_path})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)