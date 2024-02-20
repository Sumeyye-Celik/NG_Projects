# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:16:17 2023

@author: Sümeyye
"""

#speech to text with audio file

import speech_recognition as sr

# recognizer = sr.Recognizer()

# ses_dosyasi = "ses_dosyası.wav"


# with sr.AudioFile(ses_dosyasi) as source:
    
    
#     audio = recognizer.record(source)

#     try:
#         text = recognizer.recognize_google(audio)
#         print("Ses dosyasındaki metin: {}".format(text))
#     # except sr.UnknownValueError:
#     #     print("Ses anlaşılamadı")
#     except sr.RequestError as e:
#         print("Google Web Speech API hatası; {0}".format(e))


UserVoiceRecognizer = sr.Recognizer()
 
while(1):
    try:
 
        with sr.Microphone() as UserVoiceInputSource:
 
            UserVoiceRecognizer.adjust_for_ambient_noise(UserVoiceInputSource, duration=0.5)
 
            UserVoiceInput = UserVoiceRecognizer.listen(UserVoiceInputSource)
 
            UserVoiceInput_converted_to_Text = UserVoiceRecognizer.recognize_google(UserVoiceInput)
            UserVoiceInput_converted_to_Text = UserVoiceInput_converted_to_Text.lower()
            print(UserVoiceInput_converted_to_Text)
    
    except KeyboardInterrupt:
        print('A KeyboardInterrupt encountered; Terminating the Program !!!')
        exit(0)
    
    except sr.UnknownValueError:
        print("Kullanıcı sesi anlaşılmadı !!!")
