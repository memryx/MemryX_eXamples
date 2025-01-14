# Import the libraries 
import librosa
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import argparse 

from memryx import *
import numpy as np 
import soundfile as sf 

import numpy as np
import os


################################################################################################
# Helper functions - Code borrowed from: https://github.com/vbelz/Speech-enhancement/tree/master
################################################################################################

def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame):
    """This function take an audio and split into several frame
       in a numpy matrix of size (nb_frame,frame_length)"""

    sequence_sample_length = sound_data.shape[0]

    sound_data_list = [sound_data[start:start + frame_length] for start in range(
    0, sequence_sample_length - frame_length + 1, hop_length_frame)]  # get sliding windows
    sound_data_array = np.vstack(sound_data_list)

    return sound_data_array


def audio_files_to_numpy(path_to_audio_file, sample_rate, frame_length, hop_length_frame, min_duration):
    """This function take audio files of a directory and merge them
    in a numpy matrix of size (nb_frame,frame_length) for a sliding window of size hop_length_frame"""

    list_sound_array = []

    # open the audio file
    y, sr = librosa.load(path_to_audio_file, sr=sample_rate)
    total_duration = librosa.get_duration(y=y, sr=sr)

    if (total_duration >= min_duration):
        list_sound_array.append(audio_to_audio_frame_stack(y, frame_length, hop_length_frame))
    else:
        print(f"The following file {os.path.join(audio_dir,file)} is below the min duration")

    return np.vstack(list_sound_array)


def audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, audio):
    """This function takes an audio and convert into spectrogram,
       it returns the magnitude in dB and the phase"""

    stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)

    stftaudio_magnitude_db = librosa.amplitude_to_db(
        stftaudio_magnitude, ref=np.max)

    return stftaudio_magnitude_db, stftaudio_phase


def numpy_audio_to_matrix_spectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft):
    """This function takes as input a numpy audi of size (nb_frame,frame_length), and return
    a numpy containing the matrix spectrogram for amplitude in dB and phase. It will have the size
    (nb_frame,dim_square_spec,dim_square_spec)"""

    nb_audio = numpy_audio.shape[0]

    m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    m_phase = np.zeros((nb_audio, dim_square_spec, dim_square_spec), dtype=complex)

    for i in range(nb_audio):
        m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(
            n_fft, hop_length_fft, numpy_audio[i])

    return m_mag_db, m_phase


def magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, stftaudio_magnitude_db, stftaudio_phase):
    """This functions reverts a spectrogram to an audio"""

    stftaudio_magnitude_rev = librosa.db_to_amplitude(stftaudio_magnitude_db, ref=1.0)

    # taking magnitude and phase of audio
    audio_reverse_stft = stftaudio_magnitude_rev * stftaudio_phase
    audio_reconstruct = librosa.core.istft(audio_reverse_stft, hop_length=hop_length_fft, length=frame_length)

    return audio_reconstruct


def matrix_spectrogram_to_numpy_audio(m_mag_db, m_phase, frame_length, hop_length_fft)  :
    """This functions reverts the matrix spectrograms to numpy audio"""

    list_audio = []

    nb_spec = m_mag_db.shape[0]

    for i in range(nb_spec):

        audio_reconstruct = magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, m_mag_db[i], m_phase[i])
        list_audio.append(audio_reconstruct)

    return np.vstack(list_audio)


def scaled_in(matrix_spec):
    "global scaling apply to noisy voice spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec + 46)/50
    return matrix_spec


def inv_scaled_ou(matrix_spec):
    "inverse global scaling apply to noise models spectrograms"
    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec



class AudioDenoise:

    def __init__(self, model_dfp_path, path_to_save_denoised_audio, sample_rate, min_duration, frame_length, hop_length_frame, 
                       n_fft, hop_length_fft):
        """ 
        The AudioDenoise class:

        This class contains functions that read the input .wav file, preprocesses the audio (convert it to spectrograms), performs inference on the MXA 
        and does the post processing as well (convert the spectrogram to a denoised audio clip) It uses the SyncAccl for this task.
        """
        self.model_dfp_path = model_dfp_path
        self.path_to_save_denoised_audio = path_to_save_denoised_audio

        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.frame_length = frame_length
        self.hop_length_frame = hop_length_frame
        self.n_fft = n_fft
        self.hop_length_fft = hop_length_fft
        
        self.all_frames_from_input = []
        
        # Initialize the SyncAccl
        self.accl = SyncAccl(dfp = self.model_dfp_path)


    def preprocess_inputs(self, path_to_audio_file):
        """
        This functions reads the audio file, converts it to a numpy array and generates the spectrogram
        """

        # Extracting audio from file and convert to numpy
        audio = audio_files_to_numpy(path_to_audio_file, self.sample_rate, self.frame_length, self.hop_length_frame, self.min_duration)

        # Dimensions of squared spectrogram
        dim_square_spec = int(self.n_fft / 2) + 1

        # Create Amplitude and phase of the sounds
        self.m_amp_db_audio,  self.m_pha_audio = numpy_audio_to_matrix_spectrogram(audio, dim_square_spec, self.n_fft, self.hop_length_fft)

        # Global scaling to have distribution -1/1
        x_in = scaled_in(self.m_amp_db_audio)

        # Reshape input for prediction
        x_in = x_in.reshape(x_in.shape[0], x_in.shape[1], x_in.shape[2],1)

        self.total_frames_in_x_in = x_in.shape[0]

        return x_in

    
    def get_predictions(self):
        """ 
        This function gets the predictions from the MXA after inference and generates the audio file from the predicted spectrogram. The audio file is 
        subsequently saved for the user to listen to it later. 
        """
        self.output_predictions = np.stack(self.all_frames_from_input)
        inv_sca_X_pred = inv_scaled_ou(self.output_predictions)
        
        X_denoise = self.m_amp_db_audio - inv_sca_X_pred[:,:,:,0] 

        # Reconstruct audio from denoised spectrogram and phase
        audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, self.m_pha_audio, self.frame_length, self.hop_length_fft)
        
        # Number of frames
        nb_samples = audio_denoise_recons.shape[0]
        
        # Save all frames in one file
        denoise_long = audio_denoise_recons.reshape(1, nb_samples * self.frame_length)*10
        sf.write(self.path_to_save_denoised_audio, denoise_long[0, :], self.sample_rate)


    def postprocess_outputs(self, *mxa_output):
        """ 
        Collect the outputs from the MXA and do the prediction once all frames have been processed.
        """
        self.all_frames_from_input.append(mxa_output[0])

        if len(self.all_frames_from_input) == self.total_frames_in_x_in:
            self.get_predictions()


    def run(self, path_to_audio_file):
        """ 
        The run function combines all processes together. It takes in as input the path to the audio file, does the preprocessing and then calls the SyncAccl 
        for inference after which it generates the output 
        """
        # Do the preprocessing on the input
        x_in = self.preprocess_inputs(path_to_audio_file)

        for i in range(x_in.shape[0]):

            x_in_frame = x_in[i]
            x_in_frame = x_in_frame.reshape(x_in_frame.shape[0], x_in_frame.shape[1], x_in_frame.shape[2],1)
            x_in_frame = x_in_frame.astype(np.float32)

            # Perform inference on the SyncAccl
            mxa_output = self.accl.run(x_in_frame)

            self.all_frames_from_input.append(mxa_output)

            # Do the prediction once all frames have been processed
            if len(self.all_frames_from_input) == self.total_frames_in_x_in:
                self.get_predictions()


def main():

    # Create the parser
    parser = argparse.ArgumentParser(description='Speech enhancement/denoising via command line for a noisy .wav file')

    parser.add_argument('--path_to_noisy_audio_file', default='../../assets/test/noisy_voice_long_t1.wav', type=str)
    parser.add_argument('--path_to_save_denoised_audio_file', default='../../assets/save_predictions/denoise_audio.wav', type=str)

    args = parser.parse_args()

    # Get the args
    path_to_audio_file = args.path_to_noisy_audio_file
    path_to_save_denoised_audio = args.path_to_save_denoised_audio_file

    model_dfp_path = '../../models/audio_denoise/audio_denoise.dfp'

    # Using the values from https://github.com/vbelz/Speech-enhancement/tree/master
    sample_rate = 8000
    min_duration = 1.0
    frame_length = 8064
    hop_length_frame = 8064
    n_fft = 255
    hop_length_fft = 63

    # Call the AudioDenoise class 
    audiodenoise = AudioDenoise(model_dfp_path, path_to_save_denoised_audio, sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft)

    # Perform inference 
    audiodenoise.run(path_to_audio_file)

    print('Denoising complete!')
    print('The audio file is present at: ', path_to_save_denoised_audio)
       


if __name__ == '__main__':
    main()