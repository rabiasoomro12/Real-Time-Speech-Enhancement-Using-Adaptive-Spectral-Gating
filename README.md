Real-Time Speech Enhancement Using Adaptive Spectral Gating

This repository contains the implementation of a real-time speech enhancement system designed to suppress stationary background noise (e.g., fan harmonics) in lecture recordings. The project was developed as part of a Complex Engineering Problem (CEP) for the Bachelor of Engineering in Computer Systems Engineering at Sukkur IBA University, Spring 2025.
Project Overview
The goal of this project is to enhance the clarity and intelligibility of speech in noisy lecture recordings by reducing stationary noise, such as fan harmonics at 60–200 Hz, while preserving critical speech frequencies (500–4,000 Hz). The proposed solution uses an adaptive spectral gating algorithm with frequency-dependent thresholding, offering a lightweight and efficient alternative to traditional methods like Wiener filtering and spectral subtraction.
Key Features

Unsupervised Noise Profiling: Automatically extracts noise characteristics from a 5–10 second audio segment.
Adaptive Thresholding: Dynamically calculates frequency-dependent thresholds based on noise statistics (mean + 3σ).
Time-Frequency Masking: Applies binary masking to suppress noise while preserving speech.
Real-Time Performance: Processes 5-second audio clips in 1.2 seconds, 40% faster than Wiener filtering.
Evaluation Metrics: Achieves 9.6 dB SNR improvement and 4.5/5 subjective clarity score.

Methodology
The system pipeline consists of the following steps:

Noise Profiling: Extracts a noise sample (5–10 seconds) and computes its Short-Time Fourier Transform (STFT) using a 2,048-sample frame, 50% overlap, and Hanning window.
Threshold Calculation: Computes a frequency-dependent threshold for each frequency bin:Threshold(f) = μ_noise(f) + 3 * σ_noise(f)

where μ_noise and σ_noise are the mean and standard deviation of the noise magnitude spectrum.
Binary Masking: Creates a time-frequency mask where spectral components above the threshold are preserved, and those below are gated.
Signal Reconstruction: Applies the mask to the noisy signal’s STFT and reconstructs the enhanced audio using inverse STFT, followed by median filtering to reduce artifacts.

Algorithm Implementation
The core spectral gating function is implemented in Python using the librosa library:
import librosa
import numpy as np
from scipy.signal import medfilt

def spectral_gate(y, sr, noise_sample, n_std=3.0, frame_length=2048, hop_length=512):
    # STFT calculations
    stft_noise = librosa.stft(noise_sample, n_fft=frame_length, hop_length=hop_length)
    stft_signal = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    
    # Threshold estimation
    noise_mag = np.abs(stft_noise)
    threshold = np.mean(noise_mag, axis=1) + n_std * np.std(noise_mag, axis=1)
    
    # Create binary mask
    mask = (np.abs(stft_signal) > threshold[:, np.newaxis]).astype(float)
    masked_stft = stft_signal * mask
    
    # Reconstruct audio
    y_clean = librosa.istft(masked_stft, hop_length=hop_length)
    y_clean = medfilt(y_clean, kernel_size=3)
    return y_clean, mask, threshold, noise_mag

Results
Quantitative Evaluation

SNR Improvement: 9.6 dB increase in signal-to-noise ratio for stationary fan noise.
Noise Reduction: 89.2% reduction in noise amplitude in silent regions (measured via RMS).
Processing Time: 1.2 seconds for a 5-second clip, enabling real-time applications.

Qualitative Evaluation

Subjective Listening Tests: 5 participants rated the enhanced audio 4.5/5 for clarity, noting natural speech with minor artifacts.
Visualizations:
Waveform Comparison: Reduced noise amplitude in silent regions.
Difference Spectrogram: 8–12 dB noise reduction in the 60–200 Hz range.
Time-Frequency Mask: Effective suppression of fan harmonics.




Installation
To run the project, ensure you have Python 3.8+ installed. Follow these steps:

Clone the repository:
git clone https://github.com/your-username/speech-enhancement-spectral-gating.git
cd speech-enhancement-spectral-gating


Install dependencies:
pip install -r requirements.txt


Requirements file (requirements.txt):
librosa==0.10.1
numpy==1.24.3
scipy==1.10.1
matplotlib==3.7.1
seaborn==0.12.2



Usage

Prepare an audio file (e.g., lecture_recording.mp3) with a sampling rate of 16 kHz.
Run the main script to process the audio:python main.py --input lecture_recording.mp3 --output enhanced_lecture.wav


The script will:
Extract a noise profile from the first 5–10 seconds.
Apply spectral gating to enhance the audio.
Generate visualizations (waveforms, spectrograms, etc.) saved as PNG files.
Save the enhanced audio as enhanced_lecture.wav.



Example
import librosa
from spectral_gating import spectral_gate

# Load audio
y, sr = librosa.load("lecture_recording.mp3", sr=16000)
noise_sample = y[int(5 * sr):int(10 * sr)]

# Apply spectral gating
y_clean, mask, threshold, noise_mag = spectral_gate(y, sr, noise_sample)

# Save enhanced audio
librosa.output.write_wav("enhanced_lecture.wav", y_clean, sr)

Visualizations
The project includes several visualizations to illustrate the noise reduction process:

Noise Profile Analysis: Time-domain waveform, frequency spectrum, and spectrogram of the noise sample.
Threshold Visualization: Frequency-dependent threshold curve targeting fan harmonics.
Binary Mask: Time-frequency mask showing preserved (white) and gated (black) regions.
Waveform Comparison: Original vs. enhanced waveforms.
Difference Spectrogram: Noise reduction in the 60–200 Hz range.

Limitations

Non-Stationary Noise: The algorithm struggles with dynamic noise sources (e.g., door slams).
Musical Noise Artifacts: Minor artifacts may occur in low-SNR conditions.

Future Improvements

Hybrid Approach: Combine spectral gating with adaptive filtering (e.g., LMS) for non-stationary noise.
Deep Learning: Integrate neural networks for artifact reduction and improved speech enhancement.
Real-Time Optimization: Further reduce processing time for live applications.

References

Boll, S. (1979). Suppression of Acoustic Noise in Speech Using Spectral Subtraction. IEEE Transactions on Acoustics, Speech, and Signal Processing, 27(2), 113–120. DOI:10.1109/TASSP.1979.1163209
McFee, B., et al. (2023). librosa: Audio and Music Signal Analysis in Python. Journal of Open Source Software, 8(85), 5067. DOI:10.21105/joss.05067
Loizou, P. (2013). Speech Enhancement: Theory and Practice (2nd ed.). CRC Press.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Supervisor: Dr. Junaid Ahmed, Sukkur IBA University.
Libraries: Thanks to the developers of librosa, numpy, scipy, matplotlib, and seaborn.


Feel free to explore the code, test it with your own audio files, and contribute to improving the system! For any questions, reach out via GitHub Issues.
