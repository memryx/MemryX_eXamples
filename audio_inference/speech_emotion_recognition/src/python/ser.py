# Import required libraries
import tensorflow as tf 
import os
import numpy as np 
from memryx import SyncAccl
from collections import Counter

# Input Config
NUM_SPECTROGRAM_BINS = 513
NUM_MEL_BINS = 128
LOWER_EDGE_HERTZ = 80.0
UPPER_EDGE_HERTZ = 7600.0
SAMPLE_RATE = 16000
FRAME_LENGTH = 1024
FRAME_STEP = 256
FFT_LENGTH=1024
N_MFCC = 40
segment_length = 48000


# List of emotions that the model can detect (Model trained on EMO-DB dataset)
labels_list = ['happiness', 'neutral', 'anger', 'anxiety_fear', 'boredom', 'disgust', 'sadness']

# Matrix for linear scale spectrograms to mel scale
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(NUM_MEL_BINS, NUM_SPECTROGRAM_BINS, SAMPLE_RATE, LOWER_EDGE_HERTZ, UPPER_EDGE_HERTZ)


#####################################################################################################################################################
# Helper functions: Code Reference: https://github.com/AryaAftab/LIGHT-SERNET/tree/master
#####################################################################################################################################################

class SpeechEmotionRecognition:
    """
    The SpeechEmotionRecognition class contains functions that read an audio file, process it and perform inference using the MXA 
    """

    def __init__(self, model_dfp_path):

        # Initiate the SyncAccl
        self.syncaccl = SyncAccl(dfp = model_dfp_path) 


    def _reset_elements(self):

        #Reset all values before running inference
        self.intermediate_class_predictions = []


    def _get_spectrogram(self, waveform):

        # Create spectrogram
        waveform = tf.cast(waveform, tf.float32)
        spectrogram = tf.signal.stft(
            waveform, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, fft_length=FFT_LENGTH)

        spectrogram = tf.abs(spectrogram)

        return spectrogram


    def _get_mel_spectrogram(self, spectrogram):

        # Create the mel spectrogram
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

        return log_mel_spectrogram


    def _get_mfcc(self, log_mel_spectrograms, clip_value=10):

        # Calculate the MFCC
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :N_MFCC]
        return tf.clip_by_value(mfcc, -clip_value, clip_value)


    def _trim_input(self, audio_data, segment_length):

        # Trim the inputs
        if audio_data.shape[0] <= segment_length:
            return audio_data

        else:
            trim_size = audio_data.shape[0] - segment_length
            pre = int(np.floor(trim_size/2))
            pos = int(-np.floor(-trim_size/2))
        
            trimed_data = audio_data[pre:-pos]
            return trimed_data 


    def _pad_input(self, audio_data):

        # Pad input to ensure the shape matches
        padding = segment_length - np.shape(audio_data)[0]
        pre = int(np.floor(padding/2))
        pos = int(-np.floor(-padding/2))

        padded_data = np.concatenate([np.zeros(pre, dtype=np.float32), audio_data, np.zeros(pos, dtype=np.float32)], axis=0)

        return padded_data


    def _get_input_to_model(self, audio):

        # Combines all functions that take as input the audio and return the MFCC which can then be fed as input to the model
        spectrogram = self._get_spectrogram(audio)
        mel_spectrogram = self._get_mel_spectrogram(spectrogram)
        mfcc = self._get_mfcc(mel_spectrogram)
        mfcc = tf.expand_dims(mfcc, -1)
        mfcc = tf.expand_dims(mfcc, 0)
        return np.array(mfcc)


    def _normalize(self, data):

        # Normalize the data
        EPS = np.finfo(float).eps
        samples_99_percentile = np.percentile(np.abs(data), 99.9)
        normalized_samples = data / (samples_99_percentile + EPS)
        normalized_samples = np.clip(normalized_samples, -1, 1)
        normalized_samples = self._pad_input(normalized_samples)

        return normalized_samples


    def get_classification_result(self):

        # Return the result predicted by the model
        return self.counts.most_common(1)[0][0]


    def get_intermediate_class_predictions(self):

        # Calculate the total count of each predicted intermediate class and return it as a dictionary
        return dict(self.counts)

    
    def run(self, wave_path):
        
        # The run function combines all processes that read the data from the audio file, perform inference using the SyncAccl and print the prediction
        audio_binary = tf.io.read_file(wave_path)
        raw_data, sample_rate = tf.audio.decode_wav(audio_binary)
        raw_data = np.squeeze(raw_data.numpy())

        self._reset_elements()

        for index in range(0, len(raw_data), segment_length):

            trimmed_data = raw_data[index : index+segment_length]
            normalized_data = self._normalize(trimmed_data)
            mfcc = self._get_input_to_model(normalized_data)
            mxa_output = self.syncaccl.run(mfcc)
            prediction_index = np.argmax(mxa_output, axis=1)
            prediction = labels_list[prediction_index[0]]
           
            self.intermediate_class_predictions.append(prediction)

        self.counts = Counter(self.intermediate_class_predictions)

