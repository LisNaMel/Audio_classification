# """PyAudio Example: Play a wave file (callback version)."""
#
# import pyaudio
# import wave
# import time
# import sys
#
# if len(sys.argv) < 2:
#     print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
#     sys.exit(-1)
#
# wf = wave.open(sys.argv[1], 'rb')
#
# # instantiate PyAudio (1)
# p = pyaudio.PyAudio()
#
# # define callback (2)
# def callback(in_data, frame_count, time_info, status):
#     data = wf.readframes(frame_count)
#     return (data, pyaudio.paContinue)
#
# # open stream using callback (3)
# stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                 channels=wf.getnchannels(),
#                 rate=wf.getframerate(),
#                 output=True,
#                 stream_callback=callback)
#
# # start the stream (4)
# stream.start_stream()
#
# # wait for stream to finish (5)
# while stream.is_active():
#     time.sleep(0.1)
#
# # stop stream (6)
# stream.stop_stream()
# stream.close()
# wf.close()
#
# # close PyAudio (7)
# p.terminate()

# import pyaudio
# import wave
#
# FORMAT = pyaudio.paInt16
# CHANNELS = 2
# RATE = 44100
# CHUNK = 1024
# RECORD_SECONDS = 5
# WAVE_OUTPUT_FILENAME = "file.wav"
#
# audio = pyaudio.PyAudio()
#
# # start Recording
# stream = audio.open(format=FORMAT, channels=CHANNELS,
#                     rate=RATE, input=True,
#                     frames_per_buffer=CHUNK)
# print("recording...")
# frames = []
#
# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data)
# print("finished recording")
#
# # stop Recording
# stream.stop_stream()
# stream.close()
# audio.terminate()
#
# waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# waveFile.setnchannels(CHANNELS)
# waveFile.setsampwidth(audio.get_sample_size(FORMAT))
# waveFile.setframerate(RATE)
# waveFile.writeframes(b''.join(frames))
# waveFile.close()

# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.io import wavfile
#
# sample_rate, samples = wavfile.read('file.wav')
# frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
#
# plt.pcolormesh(times, frequencies, spectrogram)
# plt.imshow(spectrogram)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

# from scipy.io import wavfile
# from scipy import signal
# import numpy as np
#
# sample_rate, audio = wavfile.read('file.wav')
#
# def log_specgram(audio, sample_rate, window_size=20,
#                  step_size=10, eps=1e-10):
#     nperseg = int(round(window_size * sample_rate / 1e3))
#     noverlap = int(round(step_size * sample_rate / 1e3))
#     freqs, times, spec = signal.spectrogram(audio,
#                                     fs=sample_rate,
#                                     window='hann',
#                                     nperseg=nperseg,
#                                     noverlap=noverlap,
#                                     detrend=False)
#     return freqs, times, np.log(spec.T.astype(np.float32) + eps)

# import matplotlib.pyplot as plot
#
# from scipy.io import wavfile
#
# # Read the wav file (mono)
#
# samplingFrequency, signalData = wavfile.read('file.wav')
#
# # Plot the signal read from wav file
#
# plot.subplot(211)
#
# plot.title('Spectrogram of a wav file with piano music')
#
# plot.plot(signalData)
#
# plot.xlabel('Sample')
#
# plot.ylabel('Amplitude')
#
# plot.subplot(212)
#
# plot.specgram(signalData, Fs=samplingFrequency)
#
# plot.xlabel('Time')
#
# plot.ylabel('Frequency')
#
# plot.show()

# import matplotlib.pyplot as plot
#
# import numpy as np
#
# # Define the list of frequencies
#
# frequencies = np.arange(5, 105, 5)
#
# # Sampling Frequency
#
# samplingFrequency = 400
#
# # Create two ndarrays
#
# s1 = np.empty([0])  # For samples
#
# s2 = np.empty([0])  # For signal
#
# # Start Value of the sample
#
# start = 1
#
# # Stop Value of the sample
#
# stop = samplingFrequency + 1
#
# for frequency in frequencies:
#     sub1 = np.arange(start, stop, 1)
#
#     # Signal - Sine wave with varying frequency + Noise
#
#     sub2 = np.sin(2 * np.pi * sub1 * frequency * 1 / samplingFrequency) + np.random.randn(len(sub1))
#
#     s1 = np.append(s1, sub1)
#
#     s2 = np.append(s2, sub2)
#
#     start = stop + 1
#
#     stop = start + samplingFrequency
#
# # Plot the signal
#
# plot.subplot(211)
#
# plot.plot(s1, s2)
#
# plot.xlabel('Sample')
#
# plot.ylabel('Amplitude')
#
# # Plot the spectrogram
#
# plot.subplot(212)
#
# powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(s2, Fs=samplingFrequency)
#
# plot.xlabel('Time')
#
# plot.ylabel('Frequency')
#
# plot.show()

# from scipy.io import wavfile
# from scipy import signal
# import numpy as np
#
# sample_rate, audio = wavfile.read('file.wav')
#
# def log_specgram(audio, sample_rate, window_size=20,
#                  step_size=10, eps=1e-10):
#     nperseg = int(round(window_size * sample_rate / 1e3))
#     noverlap = int(round(step_size * sample_rate / 1e3))
#     freqs, times, spec = signal.spectrogram(audio,
#                                     fs=sample_rate,
#                                     window='hann',
#                                     nperseg=nperseg,
#                                     noverlap=noverlap,
#                                     detrend=False)
#     return freqs, times, np.log(spec.T.astype(np.float32) + eps)

# import numpy
# import scipy.io.wavfile
# from scipy.fftpack import dct
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
#
# sample_rate, signal = scipy.io.wavfile.read('file.wav')
# signal = signal[0:int(2*sample_rate)]
# emphasized_signal = numpy.append(signal[0], signal[1:] - 0.97 * signal[:-1])
# frame_size = 0.025
# frame_stride = 0.01
# frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
# signal_length = len(emphasized_signal)
# frame_length = int(round(frame_length))
# frame_step = int(round(frame_step))
# num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))
#
# pad_signal_length = num_frames*frame_step + frame_length
# z = numpy.zeros((pad_signal_length - signal_length))
# pad_signal = numpy.append(emphasized_signal, z)
#
# indices = numpy.tile(numpy.arange(0,frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
# frames = pad_signal[indices.astype(numpy.int32, copy=False)]
# frames *= numpy.hamming(frame_length)
#
# NFFT = 512
# mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))
# pow_frames = ((1.0/NFFT) * ((mag_frames)**2))
#
#
# nfilt = 40
# low_freq_mel = 0
# high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
# mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
# hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
# bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)
#
# fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
# for m in range(1, nfilt + 1):
#     f_m_minus = int(bin[m - 1])   # left
#     f_m = int(bin[m])             # center
#     f_m_plus = int(bin[m + 1])    # right
#
#     for k in range(f_m_minus, f_m):
#         fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
#     for k in range(f_m, f_m_plus):
#         fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
# filter_banks = numpy.dot(pow_frames, fbank.T)
# filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
# filter_banks = 20 * numpy.log10(filter_banks)  # dB
#
# num_ceps = 12
# mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
#
# cep_lifter = 22
# (nframes, ncoeff) = mfcc.shape
# n = numpy.arange(ncoeff)
# lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
# mfcc *= lift
#
# mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
# # print(mfcc)
# # plt.imshow(numpy.flipud(filter_banks.T), cmap=cm.jet, aspect=0.2, extent=[0,0,0,4])
#
#
#
# plt.subplot(312)
# filter_banks -= (numpy.mean(filter_banks,axis=0) + 1e-8)
# plt.imshow(filter_banks.T, cmap=cm.jet , aspect='auto')
# plt.xticks(numpy.arange(0, (filter_banks.T).shape[1],
#                         int((filter_banks.T).shape[1]/8)),
#            ['0','0.25','0.5','0.75', '1', '1.25', '1.5', '1.75','2'])
# ax = plt.gca()
# ax.invert_yaxis()
# plt.title('spectrogram')
# plt.savefig('spec_.jpeg')
# plt.clf()

# plt.imshow(mfcc)
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# plt.show()
# plt.clf()


import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed
import pylab
import librosa
import librosa.display
import numpy as np

sig, fs = librosa.load('file.wav')
# make pictures name
save_path = 'test.jpg'

pylab.axis('off') # no axis
pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
S = librosa.feature.melspectrogram(y=sig, sr=fs)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
pylab.close()

# import librosa
# y, sr = librosa.load(librosa.util.example_audio_file(),
#                       duration=5.0)
# librosa.output.write_wav('file_trim_5s.wav', y, sr)