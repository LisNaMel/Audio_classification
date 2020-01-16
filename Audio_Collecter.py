import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty

import pylab
import librosa.display
import numpy as np

from scipy.io import wavfile as w
import matplotlib.pyplot as plt
import shutil
import os
import pyaudio
import wave
import re
import pandas as pd
import save_spec

USER_INFO = []

data_csv = pd.read_csv('train.csv')
data_frame = pd.DataFrame(data_csv, columns=['wav_path', 'img_path', 'spec_path', 'spec_libro_path',
                                             'sound', 'length', 'gender', 'age'])


def sent_user_info(info):
    global USER_INFO
    USER_INFO = info


def get_data_frame():
    return data_frame


class UserInfoScreen(Screen):

    userId = ObjectProperty(None)
    gender = ObjectProperty(None)
    age = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.userInfo = []

    def submit(self):
        if self.userId.text != '' and self.gender.text != '' and self.age.text != '':
            self.userInfo = [self.userId.text, self.gender.text, self.age.text]
            sent_user_info(self.userInfo)
            self.reset()
            self.manager.current = 'Record'
        else:
            self.alert()

    def reset(self):
        self.userId.text = ""
        self.gender.text = ""
        self.age.text = ""

    def alert(self):
        print('err')
        self.reset()


class RecordScreen(Screen):

    start_stop_button = ObjectProperty(None)
    sound_img = ObjectProperty(None)
    char_img = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = 0
        self.check_count = 0
        self.data_frame = get_data_frame()
        self.sound_id = ''

    def start_stop(self):
        if self.start_stop_button.text == 'Start':
            self.start_recording()
            self.start_stop_button.text = 'Start'

    def start_recording(self):
        print(self.sound_id)
        _format = pyaudio.paInt16
        _channel = 2
        _rate = 44100
        _chunk = 1024
        _record_second = 2
        _output_filename = 'file.wav'

        audio = pyaudio.PyAudio()
        stream = audio.open(format=_format, channels=_channel,
                            rate=_rate, input=True,
                            frames_per_buffer=_chunk)

        print('recording')
        frames = []
        for i in range(0, int(_rate / _chunk * _record_second)):
            data = stream.read(_chunk)
            frames.append(data)
        print('finished')

        stream.stop_stream()
        stream.close()
        audio.terminate()

        wave_file = wave.open(_output_filename, 'wb')
        wave_file.setnchannels(_channel)
        wave_file.setsampwidth(audio.get_sample_size(_format))
        wave_file.setframerate(_rate)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()

        self.save_img()

    def save_img(self):
        lis = self.char_img.source[:]
        lis = re.split("[./]", lis)
        self.sound_id = lis[1][-2:]
        audio = 'file.wav'
        sfq, sound = w.read(audio)
        sound = sound / (2 ** 15)

        s_point = float(sound.shape[0])
        sound_one_channel = sound[:, 0]
        time_array = np.arange(0, s_point, 1)
        time_array = time_array * 0.0022

        plt.plot(time_array, sound_one_channel, color='b')
        plt.xlabel('Time(ms)')
        plt.ylabel('Amplitude')
        current_img = str(USER_INFO[0]) + str(self.sound_id) + '.jpeg'
        plt.savefig(current_img)
        plt.clf()

        save_spec.save_spec(self.sound_id)
        save_spec.save_libro_spec(self.sound_id)

    def next_sound(self):
        lis = self.char_img.source[:]
        lis = re.split("[./]", lis)
        self.sound_id = lis[1]
        if self.sound_id == 'cat':
            self.char_img.source = 'charImg/01.jpg'
        elif self.sound_id == '89':
            self.save_file(self.sound_id)
            self.char_img.source = 'charImg/cat.jpg'
        elif int(self.sound_id) < 9:
            source = 'charImg/0' + str((int(self.sound_id)+1)) + '.jpg'
            self.char_img.source = source
            print(self.char_img.source)
            self.save_file(self.sound_id)
        else:
            source = 'charImg/' + str((int(self.sound_id) + 1)) + '.jpg'
            self.char_img.source = source
            print(self.char_img.source)
            self.save_file(self.sound_id)

    def save_file(self, sound):
        init_path = '/Humanoid_Lab/Audio_classification/'
        wav_path = init_path + 'wavfiles/' + str(USER_INFO[0]) + str(sound) + '.wav'
        img_path = init_path + 'wavImgs/' + str(USER_INFO[0]) + str(sound) + '.jpeg'
        spec_path = init_path + 'spec/' + 'spec_' + str(USER_INFO[0]) + str(sound) + '.jpeg'
        spec_libro_path = init_path + 'spec_libro/' + 'spec_libro_' + str(USER_INFO[0]) + str(sound) + '.jpeg'

        shutil.copy(init_path + 'file.wav', wav_path)
        shutil.copy(init_path + str(USER_INFO[0]) + str(sound) + '.jpeg', img_path)
        shutil.copy(init_path + str(USER_INFO[0]) + 'spec_' + str(sound) + '.jpeg', spec_path)
        shutil.copy(init_path + str(USER_INFO[0]) + 'spec_libro_' + str(sound) + '.jpeg', spec_libro_path)
        print('saved')
        self.sound_id += str(int(self.sound_id)+1)

        df = pd.DataFrame([[wav_path, img_path, spec_path, spec_libro_path, sound, 2, USER_INFO[1], USER_INFO[2]]],
                          columns=['wav_path', 'img_path', 'spec_path', 'spec_libro_path', 'sound',
                                   'length', 'gender', 'age'])

        self.data_frame = self.data_frame.append(df, ignore_index=True)
        self.data_frame.to_csv('train.csv')
        print('csv_saved')

    def check(self):
        if self.sound_id:
            self.show_img()
        audio_chunk = 1024
        wav_file = wave.open('/Humanoid_Lab/Audio_classification/file.wav', 'rb')
        audio = pyaudio.PyAudio()

        stream = audio.open(format=audio.get_format_from_width(wav_file.getsampwidth()),
                            channels=wav_file.getnchannels(),
                            rate=wav_file.getframerate(),
                            output=True)

        data = wav_file.readframes(audio_chunk)

        while data:
            stream.write(data)
            data = wav_file.readframes(audio_chunk)

        stream.stop_stream()
        stream.close()
        audio.terminate()

    def show_img(self):
        audio = 'file.wav'
        sfq, sound = w.read(audio)
        # s_data_type = sound.dtype
        sound = sound / (2 ** 15)

        s_points = float(sound.shape[0])
        sound_one_channel = sound[:, 0]
        time_array = np.arange(0, s_points, 1)
        time_array = time_array * 0.0022

        plt.plot(time_array, sound_one_channel, color='b')
        plt.xlabel('Time(ms)')
        plt.ylabel('Amplitude')
        current_img = 'show' + str(USER_INFO[0]) + str(self.check_count) + '.jpeg'
        plt.savefig(current_img)
        plt.clf()
        self.sound_img.source = current_img
        self.check_count += 1

    def remove_file(self):
        filename = str(USER_INFO[0]+str(self.sound_id))
        if os.path.exists('/Humanoid_Lab/Audio_classification/wavfiles/' + filename + '.wav'):
            os.remove('/Humanoid_Lab/Audio_classification/wavfiles/' + filename + '.wav')

        if os.path.exists('/Humanoid_Lab/Audio_classification/wavImgs/' + filename + '.jpeg'):
            os.remove('/Humanoid_Lab/Audio_classification/wavImgs/' + filename + '.jpeg')

        if os.path.exists('/Humanoid_Lab/Audio_classification/spec/spec_' + filename + '.jpeg'):
            os.remove('/Humanoid_Lab/Audio_classification/spec/spec_' + filename + '.jpeg')

        if os.path.exists('/Humanoid_Lab/Audio_classification/spec_libro/spec_libro_' + filename + '.jpeg'):
            os.remove('/Humanoid_Lab/Audio_classification/spec_libro/spec_libro_' + filename + '.jpeg')

        if len(self.data_frame):
            self.data_frame = self.data_frame.drop(self.data_frame.index[-1])
            self.data_frame.to_csv('train.csv', index=False)
            print('file_removed')


class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("audio_collector.kv")


class AudioCollectorApp(App):
    def build(self):
        return kv


if __name__ == '__main__':
    AudioCollectorApp().run()