# Import necessary modules

from flask import Flask, render_template, request, redirect, url_for
import os
import base64
import subprocess
from classify_audio import AudioClassify
from werkzeug.utils import secure_filename
import ffmpeg


# Define paths to all required items
class_map_csv_text = '../../assets/yamnet_class_map.csv'
preprocess_model_path = '../../models/Audio_classification_YamNet_96_64_1_tflite_pre.tflite'
model_dfp_path = "../../models/Audio_classification_YamNet_96_64_1_tflite.dfp"
postprocess_model_path = '../../models/Audio_classification_YamNet_96_64_1_tflite_post.tflite'

# Initialize classification class
audioclassify = AudioClassify(class_map_csv_text, preprocess_model_path, model_dfp_path, postprocess_model_path)


# Configure Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Directory to save uploaded files
app.config['ALLOWED_EXTENSIONS'] = {'wav'}  # Supported file types

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """ 
    Function to check if allowed file type is uploaded
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def convert_to_wav(input_path, output_path):
    """ 
    When the user chooses to record audio instead of uploading a file, it needs to be converted into a wav format before it can be processed.
    """
    try:
        # Ensure input file exists
        if not os.path.exists(input_path):
            raise ValueError(f"Input file does not exist: {input_path}")
        
        # Attempt to convert the audio file
        print(f"Converting {input_path} to {output_path}...")
        ffmpeg.input(input_path).output(output_path, ar='16000', ac=1).run()
        print(f"Conversion successful: {output_path}")
        return True

    except ffmpeg._run.Error as e:  # Handle errors from ffmpeg
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False
    except ValueError as e:  # If the file doesn't exist
        print(f"ValueError: {e}")
        return False
    except Exception as e:  # Catch other unexpected errors
        print(f"Unexpected error: {e}")
        return False


# Actual process after the audio file is uploaded/ recorded 
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """ 
    Once the user has uploaded the file, the processing happens here in this function. It is read, converted to the appropriate format and inference is performed.
    After processing, return the results so it can be displayed in the web page.
    """
    if request.method == 'POST':

        # Handle uploaded file
        if 'file' in request.files:
            file = request.files['file']

            if file.filename == '':
                return 'No selected file'

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(input_path)

                # Convert to WAV format
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], "converted_audio.wav")

                if convert_to_wav(input_path, output_path):
                    os.remove(input_path)  # Remove the original file after conversion

                    # Perform inference
                    audioclassify.run(output_path)
                    result = audioclassify.get_result_of_classification()
                    intermediate_class_predictions = audioclassify.get_intermediate_class_predictions()

                    os.remove(output_path)  # Clean up converted file
                    return render_template('index.html', classification_result=result, intermediate_predictions=intermediate_class_predictions)
                
                else:
                    return 'Error converting file to WAV format'


        # Handle recorded audio data (Base64)
        audio_data = request.form.get("audio-data")
        
        if audio_data:
            try:
                header, encoded = audio_data.split(",", 1)  # Separate metadata and Base64
                audio_bytes = base64.b64decode(encoded)
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], "recording_input.webm")

                # Save the Base64-decoded audio file
                with open(input_path, "wb") as f:
                    f.write(audio_bytes)

                # Convert to WAV format
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], "recording_converted.wav")
                
                if convert_to_wav(input_path, output_path):
                    os.remove(input_path)  # Remove the original file after conversion

                    # Perform inference
                    audioclassify.run(output_path)
                    result = audioclassify.get_result_of_classification()
                    intermediate_class_predictions = audioclassify.get_intermediate_class_predictions()

                    os.remove(output_path)  # Clean up converted file
                    return render_template('index.html', classification_result=result, intermediate_predictions=intermediate_class_predictions)
                
                else:
                    return 'Error converting recorded audio to WAV format'

            except Exception as e:
                return f"Error processing recorded audio: {e}"

    # If not a POST request, just render the template without a result
    return render_template('index.html', classification_result=None, intermediate_predictions=None)


if __name__ == '__main__':
    app.run()
