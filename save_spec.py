import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.use('Agg') # No pictures displayed
import pylab
import librosa.display


def save_spec(count):
    sample_rate, signal = scipy.io.wavfile.read('file.wav')
    signal = signal[0:int(2*sample_rate)]
    emphasized_signal = numpy.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    frame_size = 0.025
    frame_stride = 0.01
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames*frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z)

    indices = numpy.tile(numpy.arange(0,frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    frames *= numpy.hamming(frame_length)

    NFFT = 512
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))
    pow_frames = ((1.0/NFFT) * ((mag_frames)**2))


    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB

    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

    cep_lifter = 22
    (nframes, ncoeff) = mfcc.shape
    n = numpy.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    mfcc *= lift

    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
    # print(mfcc)
    # plt.imshow(numpy.flipud(filter_banks.T), cmap=cm.jet, aspect=0.2, extent=[0,0,0,4])



    plt.subplot(312)
    filter_banks -= (numpy.mean(filter_banks,axis=0) + 1e-8)
    plt.imshow(filter_banks.T, cmap=cm.jet , aspect='auto')
    plt.xticks(numpy.arange(0, (filter_banks.T).shape[1],
                            int((filter_banks.T).shape[1]/8)),
               ['0','0.25','0.5','0.75', '1', '1.25', '1.5', '1.75','2'])
    ax = plt.gca()
    ax.invert_yaxis()
    plt.title('spectrogram')
    plt.savefig('spec_' + str(count) + '.jpeg')
    plt.clf()


def save_libro_spec(count):
    sig, fs = librosa.load('file.wav')
    # make pictures name
    save_path = 'spec_libro_'+ str(count) +'.jpeg'

    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(S, ref=numpy.max))
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()