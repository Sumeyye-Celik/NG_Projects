# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:08:10 2023

@author: Sümeyye
"""

from bs4 import BeautifulSoup
import requests
import re
import string
import random
import time
import numpy as np
from html.parser import HTMLParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer


class Chatbot:
    def __init__(self, url):
        self.url = url
        self.sent_tokens = []
        self.initialize()
        self.welcome()
    
    def welcome(self):
        print("Initializing ChatBot . . .")
        
        time.sleep(2)
        print('Type "bye" or "quit" or "exit" to end chat \n')
        
        time.sleep(3)
        
        karsilama = np.random.choice([
            "Welcome, I am ChatBot, here for you kind service",
            "Hey, Great day! I am your virtual assistant",
            "Hello, it's my pleasure meeting you",
            "Hi, I am a ChatBot. Let's chat!"])
        
        print("ChatBot >>  " + karsilama + "\n")

    def initialize(self):
        page = requests.get(self.url)
        soup = BeautifulSoup(page.content, 'html.parser')
        info = str(soup.find_all('p'))
        text = self.remove_tags(info)
        self.sent_tokens = nltk.sent_tokenize(text)

    def remove_tags(self, text):
        tag_remove = re.compile(r'<[^>]+>')
        return tag_remove.sub('', text)

    def LemTokens(self, tokens):
        lemmer = nltk.stem.WordNetLemmatizer()
        return [lemmer.lemmatize(token) for token in tokens]

    def LemNormalize(self, text):
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        return self.LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
    
    def greeting(self, sentence):
       GREETING_INPUTS = ("hello", "hi", "whats up", "hey")
       GREETING_RESPONSES = ["How can help you?", "You can ask what you want to know"]
       for word in sentence.split():
           if word.lower() in GREETING_INPUTS:
               return random.choice(GREETING_RESPONSES)

    def response(self, user_response):
        robot_response = ''
        self.sent_tokens.append(user_response)
        TfidfVec = TfidfVectorizer(tokenizer=self.LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(self.sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if req_tfidf == 0:
            robot_response = robot_response + "I think I need to read more about that..."
            return robot_response
        else:
            robot_response = robot_response + self.sent_tokens[idx]
            return robot_response

    def chat(self):
        flag = True
        
        while flag:
            user_response = input("User>> ").lower().strip()
    
            if user_response in ['by', 'bye', 'quit', 'exit']:
                flag = False
                print('ChatBot >>  See you soon! Bye!')
                time.sleep(1)
                print('\nQuitting ChatBot ...')
            else:
                print("ChatBot>> " + self.greeting(user_response)
                      if self.greeting(user_response) is not None else 
                      f"ChatBot: {self.response(user_response)} \n")
                self.sent_tokens.remove(user_response) if self.greeting(user_response) is None else None

        
        # while flag:
        #     user_response = input().lower()
    
        # if user_response != 'bye' and not (user_response == 'thanks' or user_response == 'thank you'):
        #     print("Spongebot: ", end="")
        #     print(self.response(user_response))
        #     self.sent_tokens.remove(user_response)
        # else:
        #     flag = False
        #     print("Spongebot: Anytime" if user_response == 'thanks' or user_response == 'thank you' else "Spongebot: Alvida")

        # while flag:
        #     user_response = input("User>> ").lower().strip()
    
        # if user_response in ['bye', 'quit', 'exit']:
        #     flag = False
        #     print('ChatBot >>  See you soon! Bye!')
        #     time.sleep(1)
        #     print('\nQuitting ChatBot ...')
        # else:
        #     print(self.response(user_response))
        #     self.sent_tokens.remove(user_response)
            


url = "https://en.wikipedia.org/wiki/Ottoman_Empire"
bot = Chatbot(url)


bot.chat()
