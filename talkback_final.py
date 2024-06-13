import os
import numpy as np
import pyaudio
import librosa
from sklearn.neighbors import KDTree
import joblib
from soundfile import SoundFile

chunk_size = 1024  
hop_length = chunk_size // 2
n_mfcc = 13
n_fft = 1024  
n_mels = 40  
sr = 44100
overlap_size = chunk_size // 2


threshold_on = 0.02
threshold_off = 0.01
smoothing_factor = 0.01

window = np.hanning(chunk_size)

def extract_features(y, sr, n_mfcc, hop_length, n_fft, n_mels):
    
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels), axis=1)

    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    fmin = librosa.note_to_hz('C3')
    fmax = librosa.note_to_hz('C6')
    bins_per_octave = 12
    pitch_bins = librosa.cqt_frequencies(n_bins=n_mels, fmin=fmin, bins_per_octave=bins_per_octave)
    f0 = np.zeros(D.shape[1])
    for i in range(D.shape[1]):
        frame = D[:, i]
        pitch_range_indices = np.where((pitch_bins >= fmin) & (pitch_bins <= fmax))
        peak_bin = pitch_range_indices[0][np.argmax(frame[pitch_range_indices])]
        f0[i] = librosa.hz_to_midi(pitch_bins[peak_bin])
    
    
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft), axis=1)
    
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft), axis=1)

    features = np.concatenate([mfcc, [np.mean(f0)], spectral_centroid, spectral_rolloff])

    return features

def build_kd_tree(corpus_folder, sr):
    features = []
    big_wave = []

    for root, _, files in os.walk(corpus_folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                with SoundFile(file_path) as f:
                    for block in f.blocks(blocksize=chunk_size, fill_value=0):
                        big_wave.append(block)

    for chunk in big_wave:
        chunk_windowed = chunk * window
        feature_vector = extract_features(chunk_windowed, sr, n_mfcc, hop_length, n_fft, n_mels)
        features.append(feature_vector)
    
    kd_tree = KDTree(np.array(features))
    return kd_tree, big_wave

def build_big_wave(corpus_folder):
    big_wave = []
    for root, _, files in os.walk(corpus_folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                with SoundFile(file_path) as f:
                    for block in f.blocks(blocksize=chunk_size, fill_value=0):
                        big_wave.append(block)
    return big_wave

def save_kd_tree(corpus_folder, kd_tree):
    kd_tree_file = os.path.join(corpus_folder, 'kd_tree.pkl')
    joblib.dump(kd_tree, kd_tree_file)

def load_kd_tree(corpus_folder):
    kd_tree_file = os.path.join(corpus_folder, 'kd_tree.pkl')
    kd_tree = joblib.load(kd_tree_file)
    return kd_tree

def find_nearest_neighbor(feature_vector, kd_tree, big_wave):
    try:
        dist, ind = kd_tree.query([feature_vector], k=1)
        return big_wave[ind[0][0]]
    except Exception as e:
        print(f"Error finding nearest neighbor: {e}")
        return None

def apply_envelope(audio, envelope):
    return audio * envelope

def stft_phase_vocoder(chunk):
    stft = librosa.stft(chunk, n_fft=chunk_size, hop_length=overlap_size, window='hann')
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    return magnitude, phase

def phase_vocoder_align(previous_chunk2, previous_chunk, current_chunk):
    
    magnitude_prev2, phase_prev2 = stft_phase_vocoder(previous_chunk2)
    magnitude_prev, phase_prev = stft_phase_vocoder(previous_chunk)
    magnitude_current, phase_current = stft_phase_vocoder(current_chunk)

    phase_dif_prev2 = phase_prev - phase_prev2
    phase_dif_current = phase_prev - phase_current

    
    phase_prev2 = phase_prev2 + phase_dif_prev2
    phase_current = phase_current + phase_dif_current

    phase_prev2 -= np.angle(np.exp(1j * phase_prev2))
    phase_current -= np.angle(np.exp(1j * phase_current))

    stft_prev2 = magnitude_prev2 * np.exp(1j * phase_prev2)
    stft_current = magnitude_current * np.exp(1j * phase_current)
    stft_prev = magnitude_prev * np.exp(1j * phase_prev)

    aligned_prev2 = librosa.istft(stft_prev2, hop_length=overlap_size, window='hann')
    aligned_current = librosa.istft(stft_current, hop_length=overlap_size, window='hann')
    aligned_prev = librosa.istft(stft_prev, hop_length=overlap_size, window='hann')
 
    return aligned_prev2, aligned_prev, aligned_current


class AudioProcessor:
    def __init__(self, kd_tree, big_wave, output_stream):
        self.kd_tree = kd_tree
        self.big_wave = big_wave
        self.output_stream = output_stream
        self.previous_rms = 0.0
        self.gate_open = False
        self.previous_chunk = np.zeros(chunk_size, dtype=np.float32)
        self.previous_chunk2 = np.zeros(chunk_size, dtype=np.float32)
        self.current_audio_chunk = np.zeros(chunk_size, dtype=np.float32)
               
    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        
        rms = np.sqrt(np.mean(np.square(audio_chunk)))

        rms = smoothing_factor * self.previous_rms + (1 - smoothing_factor) * rms
        self.previous_rms = rms

        envelope = rms
        max_envelope = np.max(envelope)
        envelope *= 3.0
        
        if self.gate_open:
            if max_envelope < threshold_off:
                self.gate_open = False
        else:
            if max_envelope > threshold_on:
                self.gate_open = True

        if self.gate_open:
            if self.current_audio_chunk is None:
                audio_chunk_windowed = audio_chunk * window

                feature_vector = extract_features(audio_chunk_windowed, sr, n_mfcc, hop_length, n_fft, n_mels)

                nearest_neighbor = find_nearest_neighbor(feature_vector, self.kd_tree, self.big_wave)
                
                self.current_audio_chunk = nearest_neighbor

                prev2, prev, current = phase_vocoder_align(self.previous_chunk2, self.previous_chunk, self.current_audio_chunk)
    
            if self.current_audio_chunk is not None:
                current /= .002 + np.max(np.abs(current))
                prev /= .002 + np.max(np.abs(prev))
                prev2 /= .002 + np.max(np.abs(prev2))
                windowed_chunk = current * window
                previous_window = prev * window
                previous_window2 = prev2 * window
                


                combined_chunk = np.zeros(chunk_size)
                left_chunk = previous_window2[overlap_size:] + previous_window[:overlap_size] 
                right_chunk = previous_window[overlap_size:] + windowed_chunk[:overlap_size]
                combined_chunk[:overlap_size] = left_chunk
                combined_chunk[overlap_size:] = right_chunk

                mixed_audio = apply_envelope(combined_chunk, envelope)

                self.previous_chunk2 = self.previous_chunk
                self.previous_chunk = self.current_audio_chunk
                self.current_audio_chunk = None

                self.output_stream.write(mixed_audio.astype(np.float32).tobytes())
                return (mixed_audio.astype(np.float32).tobytes(), pyaudio.paContinue)
        else:
            silent_chunk = np.zeros(chunk_size, dtype=np.float32)
            self.output_stream.write(silent_chunk.tobytes())
            self.current_audio_chunk = None
            return (silent_chunk.tobytes(), pyaudio.paContinue)


def main():
    corpus_folder = input("Please provide the path to the folder containing WAV files: ")
    
    kd_tree_file = os.path.join(corpus_folder, 'kd_tree.pkl')

    if os.path.exists(kd_tree_file):
        print("Loading KD-tree from files...")
        kd_tree = load_kd_tree(corpus_folder)
        big_wave = build_big_wave(corpus_folder)
        print("KD-tree loaded successfully.")
    else:
        print("Building KD-tree...")
        kd_tree, big_wave = build_kd_tree(corpus_folder, sr)
        save_kd_tree(corpus_folder, kd_tree)
        print("KD-tree built and saved successfully.")

    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']} (Input: {info['maxInputChannels']}, Output: {info['maxOutputChannels']})")

    input_device_index = int(input("Enter the input device index: "))
    output_device_index = int(input("Enter the output device index: "))

    output_stream = p.open(format=pyaudio.paFloat32,
                           channels=1,
                           rate=sr,
                           output=True,
                           output_device_index=output_device_index)

    audio_processor = AudioProcessor(kd_tree, big_wave, output_stream)

    input_stream = p.open(format=pyaudio.paFloat32,
                          channels=1,
                          rate=sr,
                          input=True,
                          input_device_index=input_device_index,
                          frames_per_buffer=chunk_size,
                          stream_callback=audio_processor.audio_callback)

    input_stream.start_stream()
    output_stream.start_stream()

    try:
        while input_stream.is_active():
            pass
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
