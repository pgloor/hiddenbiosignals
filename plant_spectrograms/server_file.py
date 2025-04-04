# server file collecting 5 second intervals of wav files from plant
# pip install flask

from flask import Flask, request, jsonify
import wave
import struct
import threading
import time
import os

app = Flask(__name__)

# Configuration
DATA_FILE = "sensor_data.hex"
WAV_OUTPUT_PREFIX = "output"
SAMPLE_RATE = 142  # Hz
SAMPLE_WIDTH = 2  # 16-bit samples (2 bytes)
NUM_CHANNELS = 1  # Mono
WRITE_INTERVAL = 5  # Process data every 10 seconds
PORT = 5001  # Port to listen on
OUTPUT_DIR = "oxo_data"

# Lock for thread-safe file access
file_lock = threading.Lock()

def hex_to_wav(input_file, output_file, output_dir, signed=True, endian="big"):
    """
    Convert hex sensor data into a properly formatted WAV file.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct full output file path
        output_file = os.path.join(output_dir, output_file)
        
        # Step 1: Read hex data
        with open(input_file, "r") as f:
            hex_data = f.read().strip()

        if not hex_data:
            print(f"No data found in {input_file}.")
            return False

        # Step 2: Convert hex to raw bytes
        raw_data = bytes.fromhex(hex_data)

        # Step 3: Interpret raw data as 16-bit samples
        if endian == "big":
            samples = [int.from_bytes(raw_data[i:i+2], byteorder="big", signed=signed)
                       for i in range(0, len(raw_data), 2)]
        elif endian == "little":
            samples = [int.from_bytes(raw_data[i:i+2], byteorder="little", signed=signed)
                       for i in range(0, len(raw_data), 2)]
        else:
            raise ValueError("Invalid endian format. Use 'big' or 'little'.")

        if not samples:
            print("No valid audio samples extracted from hex data.")
            return False

        print(f"Number of Samples: {len(samples)}")
        print(f"First 10 Samples: {samples[:10]}")

        # Step 4: Write WAV file
        with wave.open(output_file, "w") as wav_file:
            wav_file.setnchannels(NUM_CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)

            # Write samples to the WAV file
            for sample in samples:
                wav_file.writeframes(struct.pack('<h', sample))  # Little-endian 16-bit

        print(f"WAV file saved as {output_file}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


@app.route("/recording/<int:recording_id>/update", methods=["POST"])
def update_recording(recording_id):
    """
    Receive hex data from the sensor and append it to the hex data file.
    """
    data = request.data.decode("utf-8")  # Incoming hex data
    print(f"Received Data: {data[:100]}...")  # Print first 100 characters for debugging

    if data.strip():  # Ensure data is not empty
        with file_lock:
            with open(DATA_FILE, "a") as f:
                f.write(data)
        return jsonify({"status": "success"}), 200
    else:
        print("Empty data received.")
        return jsonify({"status": "no data received"}), 400

def generate_wav_files():
    """
    Background thread to process hex data and generate WAV files every 10 seconds.
    """
    while True:
        time.sleep(WRITE_INTERVAL)
        with file_lock:
            try:
                # Read the hex data
                with open(DATA_FILE, "r") as f:
                    hex_data = f.read().strip()

                if hex_data:
                    # Generate WAV file
                    timestamp = int(time.time())
                    output_file = f"{WAV_OUTPUT_PREFIX}_{timestamp}.wav"
                    success = hex_to_wav(DATA_FILE, output_file, OUTPUT_DIR, signed=True, endian="big")
                    if success:
                        # Clear the file only after successfully processing data
                        with open(DATA_FILE, "w") as f:
                            pass  # Clear the file
                    else:
                        print(f"Failed to generate WAV file for data: {hex_data[:100]}...")
                else:
                    print("No hex data to process.")
            except Exception as e:
                print(f"Error in WAV generation: {e}")


# Start the background thread
threading.Thread(target=generate_wav_files, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)

