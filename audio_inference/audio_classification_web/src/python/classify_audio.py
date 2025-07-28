# Import all the necessary modules

from ai_edge_litert.interpreter import Interpreter
import tensorflow as tf
import numpy as np
import csv
from scipy.io import wavfile
import matplotlib.pyplot as plt
import scipy
from tensorflow.keras.models import load_model
from memryx import SyncAccl
import argparse
from collections import Counter


class AudioClassify:
    """
    The AudioClassify class contains functions that read an audio file, process it before sending it to the model
    and obtaining predictions. The SyncAccl is used for this task
    """

    def __init__(self, class_map_csv_text, preprocess_model_path, model_dfp_path, postprocess_model_path):

        self.class_map_csv_text = class_map_csv_text

        self.preprocess_model_path = preprocess_model_path
        self.model_dfp_path = model_dfp_path
        self.postprocess_model_path = postprocess_model_path

        self.num_samples_per_frame = 15600
        self.hop_size = 7800
        self.labels = self._class_names_from_csv()

        # Initialize the Sync Accl
        self.accl = SyncAccl(dfp = self.model_dfp_path)

        # Initialize the interpreters
        self._pre_interpreter = Interpreter(self.preprocess_model_path)
        self._post_interpreter = Interpreter(self.postprocess_model_path)


    def _reset_elements(self):
        """
        Reset all values before running inference
        """
        self.intermediate_class_predictions = []
        self.intermediate_scores_list = []

    
    def _class_names_from_csv(self):
        """
        Code reference: https://www.kaggle.com/models/google/yamnet/tensorFlow2
        Find the name of the class with the top score when mean-aggregated across frames. Returns list of class names corresponding to score vector.
        """
        class_names = []

        with tf.io.gfile.GFile(self.class_map_csv_text) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row['display_name'])

        return class_names


    def _ensure_sample_rate(self, original_sample_rate, waveform, desired_sample_rate=16000):
        """
        Code reference: https://www.kaggle.com/models/google/yamnet/tensorFlow2
        Resample waveform to ensure that it has the correct sampling rate that is expected by the model.
        """
        if original_sample_rate != desired_sample_rate:

            desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
            waveform = scipy.signal.resample(waveform, desired_length)

        return desired_sample_rate, waveform


    def _convert_to_mono(self, waveform):
        """
        Convert the audio from stereo to mono channel.
        """
        # Check if the audio is stereo (2 channels)
        if len(waveform.shape) == 2:
            # Convert to mono by averaging the two channels
            waveform_mono = np.mean(waveform, axis=1).astype(waveform.dtype)

        else:
            # Use the audio directly
            waveform_mono = waveform

        return waveform_mono


    def _pad_waveform_if_necessary(self, waveform_frame):
        """
        If the input audio is too short, pad it with 0s to ensure the right length before passing it to the model. Otherwise,
        return the input audio as is
        """
        waveform_frame_length = waveform_frame.shape[0]
        num_missing_samples = self.num_samples_per_frame - waveform_frame_length

        if waveform_frame_length < self.num_samples_per_frame:
            padded_waveform_frame = np.pad(waveform_frame, (0, num_missing_samples), mode='constant', constant_values=0)

        else:
            padded_waveform_frame = waveform_frame
        
        return padded_waveform_frame


    def _run_preprocess_model(self, waveform):
        """
        Preprocess the data before sending it to the DFP. The preprocessing tflite model
        is used to process the data and the output obtained is then fed into the DFP
        """

        input_details = self._pre_interpreter.get_input_details()
        waveform_input_index = input_details[0]['index']

        output_details = self._pre_interpreter.get_output_details()
        preprocessed_waveform = output_details[1]['index'] # We use the output at index 1 as that is the input that is required by the DFP

        self._pre_interpreter.resize_tensor_input(waveform_input_index, [waveform.size], strict=True)
        self._pre_interpreter.allocate_tensors()
        self._pre_interpreter.set_tensor(waveform_input_index, waveform)

        self._pre_interpreter.invoke()

        preprocessed_wav_data = self._pre_interpreter.get_tensor(preprocessed_waveform)

        return preprocessed_wav_data


    def _run_postprocess_model(self, output):
        """
        Once the outputs from the MXA have been generated, run it through the post processing model to get the scores for each class
        """

        input_details = self._post_interpreter.get_input_details()
        waveform_input_index = input_details[0]['index']

        output_details = self._post_interpreter.get_output_details()
        postprocessed_waveform = output_details[0]['index'] 

        self._post_interpreter.resize_tensor_input(waveform_input_index, list(output.shape), strict=True)
        self._post_interpreter.allocate_tensors()
        self._post_interpreter.set_tensor(waveform_input_index, output)

        self._post_interpreter.invoke()

        postprocessed_wav_data = self._post_interpreter.get_tensor(postprocessed_waveform)

        return postprocessed_wav_data


    def _preprocess_inputs(self, audio_file_path):
        """
        The function that combines all preprocessing steps right from loading the audio file, ensuring it has the 
        right sampling rate, it's single channel, padded properly etc. 

        NOTE: The YAMNet model takes fixed-length frames of audio (15600 samples) and returns a single vector of 521 audio 
        event classes. 
        Since we wish to be able to do predictions on audio clips which are larger than the expected size, we use the same method 
        of window sliding and hopping that is described here: https://www.kaggle.com/models/google/yamnet/tensorFlow2
        So, we generate frames of audio from the input file and run inference on each.
        """ 

        # Load audio file
        sample_rate, wav_data = wavfile.read(audio_file_path, 'rb')

        # Preprocess the file
        sample_rate, wav_data = self._ensure_sample_rate(sample_rate, wav_data)
        wav_data = self._convert_to_mono(wav_data)
        wav_data = wav_data.astype(np.float32)

        # Display basic information about the audio
        duration = len(wav_data)/sample_rate

        print(f'Sample rate: {sample_rate} Hz')
        print(f'Total duration: {duration:.2f}s')

        # Calculate the number of frames
        self.total_num_frames_in_audio_signal = (len(wav_data) - self.num_samples_per_frame) // self.hop_size + 1
        
        # In the rare event where the audio signal is less than 0.975s, pad it
        if len(wav_data) < self.num_samples_per_frame:
            wav_data = self._pad_waveform_if_necessary(wav_data)
            self.total_num_frames_in_audio_signal = 1

        return wav_data


    def run(self, audio_file_path):
        """
        The run function that preprocesses the input, performs inference on the MXA and does the post processing on the MXA outputs
        """

        self._reset_elements()

        # Get the preprocessed wav data
        wav_data = self._preprocess_inputs(audio_file_path)

        # Get frames
        for i in range(0, len(wav_data) - self.num_samples_per_frame + 1, self.hop_size):
            
            wav_data_frame = wav_data[i : i + self.num_samples_per_frame]
            wav_data_frame = self._pad_waveform_if_necessary(wav_data_frame)
            wav_data_frame = np.array(wav_data_frame)

            # Normalize wave data in [-1.0, 1.0] (Refer to Yamnet model documentation)
            wav_data_frame = wav_data_frame / tf.int16.max
            preprocessed_wav_data = self._run_preprocess_model(wav_data_frame)

            # Get the MXA output
            mxa_output = self.accl.run(preprocessed_wav_data)
            
            # Run the output through the post processing model to generate the class scores
            scores = self._run_postprocess_model(mxa_output[1])

            # Get the prediction for each frame
            self._get_prediction(scores)


    def get_result_of_classification(self):
        """ 
        Return the result after the model has finished classifying
        """
        return self.predicted_class

    
    def get_intermediate_class_predictions(self):
        """ 
        Calculate the total count of each predicted intermediate class and return it as a dictionary
        """
        counts = Counter(self.intermediate_class_predictions)
        return dict(counts)

     
    def _get_final_prediction(self):
        """ 
        Get the final prediction averaged over all classes to find out which class the audio clip belongs to
        """
        final_scores = np.vstack(self.intermediate_scores_list)
        self.predicted_class = self.labels[final_scores.mean(axis=0).argmax()]
        

    def _get_prediction(self, scores):
        """
        Predict the class that the audio corresponds to
        """
        # Scores has a shape of (1,521)
        top_class_index = scores.argmax()
       
        self.intermediate_class_predictions.append(self.labels[top_class_index])
        self.intermediate_scores_list.append(scores)
        
        if len(self.intermediate_scores_list) == self.total_num_frames_in_audio_signal:
            self._get_final_prediction()


