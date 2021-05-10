# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:00:45 2021

@author: Linda
"""
import speech_recognition as ses_tanima

sesi_tani = ses_tanima.Recognizer()

with ses_tanima.AudioFile('deneme.wav') as kaynak:
    ses_metni = sesi_tani.listen(kaynak)
    
    try:
        
        text = sesi_tani.recognize_google(ses_metni, language='tr-TR')
        print("ses metni :")
        print(text)
        
    except:
        print("bir hata oluştu")
       
