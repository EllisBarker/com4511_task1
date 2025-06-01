import argparse
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from scipy.fftpack import dct
from scipy.signal import resample
from scipy.spatial.distance import euclidean

class MFCCComputer:
    def __init__(self, 
                 preemph: float, 
                 window_type: str,
                 frame_step_seconds: float, 
                 frame_length_seconds: float, 
                 num_filters: int, 
                 num_ceps: int, 
                 cep_lifter: int,
                 target_fs: int,
                 apply_normalisation: bool):
        """
        Create an MFCC computer with predefined parameters used in the process.

        Args:
            preemph (float): The coefficient of the pre-emphasis filter.
            window_type (str): Type of windowing function to apply to signal frames.
            frame_step_seconds(float): The amount of time between each frame (in seconds).
            frame_length_seconds(float): The amount of time in each frame (in seconds).
            num_filters (int): The number of Mel-filters applied to the signal.
            num_ceps (int): The number of cepstral coefficients to retain.
            cep_lifter (int): The order of liftering to apply to the MFCC.
            target_fs (int): The target sampling frequency to resample to.
            apply_normalisation (bool): Whether to apply cepstral mean and variance normalisation.
        """
        self.preemph = preemph
        self.window_type = window_type
        self.frame_step_seconds = frame_step_seconds
        self.frame_length_seconds = frame_length_seconds
        self.num_filters = num_filters
        self.num_ceps = num_ceps
        self.cep_lifter = cep_lifter
        self.target_fs = target_fs
        self.apply_normalisation = apply_normalisation
        self.configuration_title = configuration_title
    
    def freq2mel(self, freq: float) -> float:
        """
        Convert Frequency in Hertz to Mels

        Args:
            freq (float): A value in Hertz. This can also be a numpy array.

        Returns:
            float: A value in Mels.
        """
        return 2595 * np.log10(1 + freq / 700.0)

    def mel2freq(self, mel: float) -> float:
        """
        Convert a value in Mels to Hertz

        Args:
            mel (float): A value in Mels. This can also be a numpy array.

        Returns:
            float: A value in Hertz.
        """
        return 700 * (10 ** (mel / 2595.0) - 1)

    def read_signal(self, filename: str) -> tuple[int, np.ndarray]:
        """
        Load and store an audio file and its associated sampling frequency.

        Args:
            filename (str): The path to the audio file to compute the MFCC for.

        Returns:
            int: The sampling frequency of the audio.
            np.ndarray: The audio data from the file.
        """
        fs_hz, signal = wav.read(filename)

        # Resample the audio if it does not match the target sampling frequency
        if fs_hz != self.target_fs:
            signal = self.resample_audio(signal, fs_hz)
            fs_hz = self.target_fs

        self.set_parameters_from_sampling_frequency(fs_hz)
        return fs_hz, signal
    
    def resample_audio(self, signal: np.ndarray, fs_hz: int) -> np.ndarray:
        """
        Resample the audio to the target sampling frequency (allows comparison of MFCCs).

        Args:
            signal (np.ndarray): The audio data from the file.
            fs_hz (int): The sampling frequency of the audio.

        Returns:
            np.ndarray: The audio data after having been resampled.
        """
        signal_duration = len(signal) / fs_hz
        target_duration = int(round(signal_duration * self.target_fs))
        signal = resample(signal, target_duration)
        return signal

    def set_parameters_from_sampling_frequency(self, fs_hz: int):
        """
        Determine frame length/step size (in samples), NFFT (Non-Uniform Fast Fourier Transform)
        and frequency thresholds based on the sampling frequency of the audio.

        Args:
            fs_hz (int): The sampling frequency of the audio.
        """
        self.frame_length_samples = int(round(self.frame_length_seconds * fs_hz))
        self.frame_step_samples = int(round(self.frame_step_seconds * fs_hz))
        self.NFFT = 1<<(self.frame_length_samples-1).bit_length()
        self.low_freq_hz = 0
        self.high_freq_hz = 8000
        nyquist = fs_hz / 2.0
        if self.high_freq_hz > nyquist:
            self.high_freq_hz = nyquist
    
    def apply_preemphasis(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply pre-emphasis to the audio signal to enhance higher frequencies that improve clarity
        of certain elements of speech (e.g. consonants).

        Args:
            signal (np.ndarray): The audio data from the file.

        Returns:
            np.ndarray: The audio data after having pre-emphasis applied to it.
        """
        return np.append(signal[0], signal[1:] - self.preemph * signal[:-1])
    
    def frame_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Separate audio signal into frames (reducing impact of varying features of speech) after 
        applying zero padding to the end of it.

        Args:
            signal (np.ndarray): The audio data after having pre-emphasis applied to it.

        Returns:
            np.ndarray: The audio data separated into frames.
        """
        num_frames = int(np.ceil(float(np.abs(len(signal) - self.frame_length_samples)) / self.frame_step_samples))
        # Append zeros to the end of the signal according to the frame length
        pad_signal_length = num_frames * self.frame_step_samples + self.frame_length_samples
        pad_signal = np.append(signal, np.zeros((pad_signal_length - len(signal))))

        frames = []
        for i in range(num_frames):
            start_index = i * self.frame_step_samples
            end_index = start_index + self.frame_length_samples
            frames.append(pad_signal[start_index:end_index])
        return np.array(frames)

    def apply_hamming_window(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply Hamming window to smooth the edges of each frame (less drastic changes to amplitude
        across frames).

        Args:
            frames (np.ndarray): The audio data separated into frames.

        Returns:
            np.ndarray: The frames of audio data after having a window applied to them.
        """
        return frames * np.hamming(self.frame_length_samples)
    
    def apply_hanning_window(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply Hanning window to smooth the edges of each frame (less drastic changes to amplitude
        across frames).

        Args:
            frames (np.ndarray): The audio data separated into frames.

        Returns:
            np.ndarray: The frames of audio data after having a window applied to them.
        """
        return frames * np.hanning(self.frame_length_samples)

    def derive_power_spectrum(self, frames: np.ndarray) -> np.ndarray:
        """
        Derive the power spectrum from the frames in order to change from the time to the
        frequency domain.

        Args:
            frames (np.ndarray): The frames of audio data after having a window applied to them.

        Returns:
            np.ndarray: The power spectrum of the audio data frames.
        """
        magspec = np.absolute(np.fft.rfft(frames, self.NFFT))
        powspec = ((1.0 / self.NFFT) * ((magspec) ** 2))
        return powspec

    def apply_mel_filterbank(self, powspec: np.ndarray, fs_hz: int) -> np.ndarray:
        """
        Obtain the energy for specific frequencies (divided across 'bins') from the power spectrum.

        Args:
            powspec (np.ndarray): The power spectrum of the audio data frames
            fs_hz (int): The sampling frequency of the audio.

        Returns:
            np.ndarray: The Mel-spectrum derived from the original audio data.
        """
        low_freq_mel = self.freq2mel(self.low_freq_hz)
        high_freq_mel = self.freq2mel(self.high_freq_hz)
        points_mel = np.linspace(low_freq_mel, high_freq_mel, self.num_filters+2)
        points_hz = self.mel2freq(points_mel)
        bin = np.floor((self.NFFT + 1) * points_hz / fs_hz)
        mel_filters = np.zeros((self.num_filters, self.NFFT // 2 + 1))

        # Obtain energy in each mel-filter bin through the power spectrum
        for i in range(1, self.num_filters + 1):
            left = bin[i-1]
            centre = bin[i]
            right = bin[i+1]
            for j in range(int(left), int(centre)):
                mel_filters[i-1, j] = (j - left) / (centre - left)
            for j in range(int(centre), int(right)):
                mel_filters[i-1, j] = (right - j) / (right - centre)

        fbank = np.dot(powspec, mel_filters.T)
        fbank = np.where(fbank == 0, np.finfo(float).eps, fbank)
        fbank = 20 * np.log10(fbank)
        return fbank

    def apply_dct(self, fbank: np.ndarray) -> np.ndarray:
        """
        Apply a DCT (Discrete Cosine Transform) to the Mel-spectrum in order to highlight the most
        important aspects of the audio data in each frame.

        Args:
            fbank (np.ndarray): The Mel-spectrum derived from the original audio data.

        Returns:
            np.ndarray: The MFCC (Mel-Frequency Cepstral Coefficients) of the audio data.
        """
        # Ignore C0
        mfcc = dct(fbank, type=2, axis=1, norm='ortho')[:, 1:self.num_ceps+1]
        return mfcc

    def apply_lifter(self, mfcc: np.ndarray) -> np.ndarray:
        """
        Compute and apply a lifter to enhance certain cepstral coefficients.

        Args:
            mfcc (np.ndarray): The MFCC of the audio data.

        Returns:
            np.ndarray: The MFCC after applying the lifter.
        """
        lift = 1 + (self.cep_lifter / 2.0) * np.sin(np.pi * np.arange(self.num_ceps) / self.cep_lifter)
        mfcc *= lift
        return mfcc

    def apply_cepstral_mean_variance_normalisation(self, mfcc: np.ndarray) -> np.ndarray:
        """
        Apply cepstral mean and variance normalisation in order to aid in reducing the impact of
        external factors (such as background noise) in the audio signal.

        Args:
            mfcc (np.ndarray): The MFCC (Mel-Frequency Cepstral Coefficients) of the audio data.

        Returns:
            np.ndarray: The MFCC after applying cepstral mean and variance normalisation.
        """
        mfcc -= np.mean(mfcc, axis=1, keepdims=True)
        mfcc /= np.std(mfcc, axis=1, keepdims=True)
        return mfcc
    
    def write_to_file(self, mfcc: np.ndarray, filename: str):
        """
        Write the results (MFCC) to a file with a given name.

        Args:
            mfcc (np.ndarray): The MFCC (Mel-Frequency Cepstral Coefficients) of the audio data.
            filename (str): The file to be created to store the MFCC.
        """
        with open(filename, 'w') as f:
            for frame in mfcc:
                f.write(" ".join(map(str, frame)) + "\n")

    def compute_mfcc(self, filename: str) -> np.ndarray:
        """
        Compute the MFCC of a given audio file.

        Args:
            filename (str): The file path of the audio file.

        Returns:
            np.ndarray: The MFCC of the audio file.
        """
        fs_hz, signal = self.read_signal(filename)
        if self.preemph != None:
            signal_preemph = self.apply_preemphasis(signal)
        else:
            signal_preemph = signal
        frames = self.frame_signal(signal_preemph)
        if self.window_type == "Hamming":
            frames_windowed = self.apply_hamming_window(frames)
        elif self.window_type == "Hanning":
            frames_windowed = self.apply_hanning_window(frames)
        else:
            frames_windowed = frames
        powspec = self.derive_power_spectrum(frames_windowed)
        fbank = self.apply_mel_filterbank(powspec, fs_hz)
        mfcc = self.apply_dct(fbank)
        if self.cep_lifter != None:
            mfcc = self.apply_lifter(mfcc)
        if self.apply_normalisation:
            mfcc = self.apply_cepstral_mean_variance_normalisation(mfcc)
        return mfcc

    def display_mfcc_heatmap(self, data: np.ndarray, title: str, xlabel: str, ylabel: str):
        """
        Display a heatmap of the contents of the MFCC.

        Args:
            data (np.ndarray): The MFCC to display.
            title (str): The title to display above the heatmap.
            xlabel (str): The label for the x-axis of the heatmap.
            ylabel (str): The label for the y-axis of the heatmap.
        """
        plt.figure(1)
        plt.imshow(data, cmap="hot", aspect="auto", interpolation="nearest")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

class KNNClassifier:
    def __init__(self, 
                 class_names: list[str], 
                 training_mfccs: list[float], 
                 training_labels: list[str], 
                 testing_mfccs: list[float], 
                 testing_labels: list[str], 
                 k: int):
        """
        Create a k-nearest neighbour classifier that uses specified training/testing data.

        Args:
            class_names (list[str]): Names of classes to assign to testing data.
            training_mfccs (list[float]): Training data for the classifier (MFCC values).
            training_labels (list[str]): Training data for the classifier (class names for each MFCC).
            testing_mfccs (list[float]): Testing data for the classifier (MFCC values).
            testing_labels (list[str]): Testing data for the classifier (class names for each MFCC).
            k (int): Number of neighbours to consider in majority ruling process for prediction.
        """
        self.class_names = class_names
        self.training_mfccs = training_mfccs
        self.training_labels = training_labels
        self.testing_mfccs = testing_mfccs
        self.testing_labels = testing_labels
        self.k = k

    def classify(self) -> list[str]:
        """
        Assign labels to testing data based on euclidean distances to training MFCCs.

        Returns:
            list[str]: Labels that have been assigned to each MFCC in the testing data.
        """
        predicted_labels = []
        for test_index in range(0, len(self.testing_mfccs)):
            euclidean_distances = []
            for train_index in range(0, len(self.training_mfccs)):
                euclidean_distances.append(
                    euclidean(self.training_mfccs[train_index], self.testing_mfccs[test_index])
                )
            closest_indices = np.argsort(euclidean_distances)[:self.k]
            closest_labels = [self.training_labels[i] for i in closest_indices]
            closest_label = self.retrieve_closest_label(closest_labels)
            predicted_labels.append(closest_label)
        return predicted_labels

    def retrieve_closest_label(self, closest_labels: np.ndarray) -> str:
        """
        Count occurrences of each class label and pick the closest (ensuring determinism by basing the
        choice on the order of the class names).

        Args:
            closest_labels (np.ndarray): Array of labels of k-nearest neighbours to the data point.

        Returns:
            str: The label of the closest data point.
        """
        closest_label = None
        closest_label_count = -1

        for label in self.class_names:
            count = closest_labels.count(label)
            if count > closest_label_count:
                closest_label = label
                closest_label_count = count

        return closest_label

    def calculate_macro_f1(self, predicted_labels: list[str]) -> float:
        """
        Derive a macro-F1 score from the precision and recall of each speaker class.

        Args:
            predicted_labels (list[str]): Labels that have been assigned to each MFCC in the testing data.

        Returns:
            float: The macro-F1 score to evaluate classifier performance for all speaker classes.
        """
        f1_scores = []
        for class_name in self.class_names:
            true_positives = 0
            false_positives = 0
            false_negatives = 0

            for class_index in range(0, len(predicted_labels)):
                # The predicted speaker and true speaker are the specified speaker
                if (self.testing_labels[class_index] == predicted_labels[class_index] and 
                    self.testing_labels[class_index] == class_name):
                    true_positives += 1
                # The predicted speaker is not the specified speaker but the true speaker is the specified speaker
                if (self.testing_labels[class_index] == class_name and 
                    predicted_labels[class_index] != class_name):
                    false_negatives += 1
                # The predicted speaker is the specified speaker but the true speaker is not the specified speaker
                if (self.testing_labels[class_index] != class_name and 
                    predicted_labels[class_index] == class_name):
                    false_positives += 1

            if true_positives != 0 or false_positives != 0:
                precision = (true_positives) / (true_positives + false_positives)
            else:
                precision = 0
                
            if true_positives != 0 or false_negatives != 0:
                recall = (true_positives) / (true_positives + false_negatives)
            else:
                recall = 0
                
            if precision != 0 or recall != 0:
                f1_score = (2 * precision * recall) / (precision + recall)
            else:
                f1_score = 0
                
            f1_scores.append(f1_score)

        print (f1_scores)
        macro_f1 = 1/len(self.class_names) * sum(f1_scores)
        return macro_f1

def experiment1():
    """
    Generate an MFCC based on a single audio file.
    """
    mfcc = computer.compute_mfcc("SA1.wav")
    computer.write_to_file(mfcc, "MFCC1.txt")
    computer.display_mfcc_heatmap(mfcc.T, 
                                  "MFCC (" + configuration_title + ")", 
                                  "Frame Index", 
                                  "Cepstral Coefficient Index")

def experiment2():
    """
    Utilise MFCCs for a number of speakers in a speaker identification system (kNN classifier).
    """
    class_names = ["1_US_English_Teens_Male", 
                   "2_British_English_Thirties_Male",
                   "3_US_English_Teens_Female",
                   "4_US_English_Thirties_Male",
                   "5_Nigerian_English_Twenties_Female",
                   "6_Canadian_English_Thirties_Male",
                   "7_Canadian_English_Fifties_Female",
                   "8_US_English_Thirties_Female",
                   "9_Scottish_English_Fifties_Male",
                   "10_Turkish_English_Twenties_Male",
                   "11_British_English_Thirties_Male",
                   "12_SouthAsia_English_Twenties_Male",
                   "13_US_English_Sixties_Female",
                   "14_US_English_Twenties_Female",
                   "15_Scottish_English_Fourties_Female",
                   "16_Filipino_English_Twenties_Female"]
    mean_training_mfccs = []
    mean_training_labels = []
    mean_testing_mfccs = []
    mean_testing_labels = []
    for class_index in range(0, len(class_names)):
        for clip_index in range(1, num_clips+1):
            filename = "speakers/" + class_names[class_index] + "/" + str(class_index+1) + "_Speech" + str(clip_index) + ".wav"
            mfcc = computer.compute_mfcc(filename)
            mean_mfcc = np.mean(mfcc, axis=0)
            if clip_index <= num_training:
                mean_training_mfccs.append(mean_mfcc)
                mean_training_labels.append(class_names[class_index])
            else:
                mean_testing_mfccs.append(mean_mfcc)
                mean_testing_labels.append(class_names[class_index])
    classifier = KNNClassifier(class_names,
                               mean_training_mfccs, 
                               mean_training_labels, 
                               mean_testing_mfccs, 
                               mean_testing_labels,
                               3)
    predicted_labels = classifier.classify()
    macro_f1 = classifier.calculate_macro_f1(predicted_labels)
    print(macro_f1)

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, choices=["experiment1", "experiment2"], required=True)
parser.add_argument("--parameter_configuration", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], required=True)
args = parser.parse_args()
parameter_configuration = args.parameter_configuration
configuration_title = {0: "Baseline",
                       1: "No Hamming Window",
                       2: "40 MFCCs",
                       3: "80 Filterbanks, 40 MFCCs",
                       4: "No Pre-emphasis",
                       5: "Hanning Window",
                       6: "No Cepstral Mean Variance Normalisation",
                       7: "No Lifter",
                       8: "80 Filterbanks, 40 MFCCs, No Hamming Window",
                       9: "80 Filterbanks, 40 MFCCs, No Cepstral Mean Variance Normalisation"}[parameter_configuration]
experiment_type = args.experiment

num_clips = 9
num_training = 6

# Set the parameters for the MFCC computer based on the argument provided
if parameter_configuration == 4:
    preemph = None
else:
    preemph = 0.97

if parameter_configuration not in [1,5,8]:
    window_type = "Hamming"
elif parameter_configuration == 5:
    window_type = "Hanning"
else: 
    window_type = None

if parameter_configuration in [2,3,8,9]:
    num_ceps = 40
else:
    num_ceps = 12

if parameter_configuration in [3,8,9]:
    num_filters = 80
elif parameter_configuration == 2:
    # For 40 MFCCs and ensure lifter functionality, need to match the number of filterbanks
    num_filters = 41
else: 
    num_filters = 26

if parameter_configuration not in [6,9]:
    apply_normalisation = True
else:
    apply_normalisation = False

if parameter_configuration != 7:
    cep_lifter = 22
else:
    cep_lifter = None

frame_step_seconds = 0.010
frame_length_seconds = 0.025
target_fs = 16000
computer = MFCCComputer(preemph,
                        window_type,
                        frame_step_seconds, 
                        frame_length_seconds, 
                        num_filters, 
                        num_ceps, 
                        cep_lifter, 
                        target_fs, 
                        apply_normalisation)

if experiment_type == "experiment1":
    experiment1()
elif experiment_type == "experiment2":
    experiment2()
else:
    print ("ERROR: Invalid experiment name")
