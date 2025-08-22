DATA_LOCATION = "./data/ivr/audio"

import os
from flask import Flask, Response, stream_with_context
import time

app = Flask(__name__)

all_audio_files = os.listdir(DATA_LOCATION)

# sort the files in the directory
all_audio_files.sort()

print(all_audio_files)
current_audio_file_index = 0

def generate_audio_stream():
    global current_audio_file_index, all_audio_files
    print("Generating audio stream")
    # Open your wav file in binary mode
    
    while True:
        if current_audio_file_index >= len(all_audio_files):
            print("End of all files")
            yield b""
            time.sleep(1)  
        current_audio_file = all_audio_files[current_audio_file_index]
        full_audio_file_path = os.path.join(DATA_LOCATION, current_audio_file)
        print(f"Streaming file: {full_audio_file_path}")
        with open(full_audio_file_path, "rb") as wav:
            while True:
                chunk_size = 1024  # bytes per chunk
                chunk = wav.read(chunk_size)
                if not chunk:
                    print("End of file")
                    current_audio_file_index += 1
                    break
                yield chunk
                # Optional: simulate live streaming pace
                time.sleep(0.008)  
        

@app.route("/audio")
def stream_audio():
    # stream_with_context ensures the request context remains active
    return Response(stream_with_context(generate_audio_stream()),
                    mimetype="audio/x-wav")

if __name__ == "__main__":
    app.run(debug=True, threaded=True)