import kivy

kivy.require('2.0.0')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
model = load_model('url_threat_detection_model.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


class URLThreatDetectionApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.label = Label(text='Введите URL для анализа угроз:')
        self.layout.add_widget(self.label)

        self.text_input = TextInput(multiline=False)
        self.layout.add_widget(self.text_input)

        self.button = Button(text='Анализировать')
        self.button.bind(on_press=self.analyze_url)
        self.layout.add_widget(self.button)

        return self.layout

    def analyze_url(self, instance):
        url = self.text_input.text
        seq = tokenizer.texts_to_sequences([url])
        padded_seq = pad_sequences(seq, maxlen=100)

        prediction = model.predict(padded_seq)
        result = 'Вредоносный' if prediction[0][0] > 0.5 else 'Безопасный'

        popup = Popup(title='Результат анализа',
                      content=Label(text=f'URL: {url}\nРезультат: {result}'),
                      size_hint=(0.8, 0.8))
        popup.open()


if __name__ == '__main__':
    URLThreatDetectionApp().run()
