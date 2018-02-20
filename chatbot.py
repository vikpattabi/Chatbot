#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA6, CS124, Stanford, Winter 2018
# v.1.0.2
# Original Python code by Ignacio Cases (@cases)
######################################################################
import csv
import math

import numpy as np
from movielens import ratings
from random import randint
import re
from PorterStemmer import PorterStemmer

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    #############################################################################
    # `moviebot` is the default chatbot. Change it to your chatbot's name       #
    #############################################################################
    def __init__(self, is_turbo=False):
      self.name = 'moviebot'
      self.is_turbo = is_turbo
      self.read_data()
      self.likes_vec = np.zeros(len(self.titles))
      self.errors = self.readInFile('data/errors.txt')
      self.articles = ['The', 'A', 'An']
      self.binarize()
      self.datapoints = 0
      self.gaveRecommendation = False
      self.INFO_THRESHOLD = 5
      self.stemmer = PorterStemmer()
      self.titles_map = self.processTitles(self.titles)

    #############################################################################
    # 1. WARM UP REPL
    #############################################################################

    def greeting(self):
      """chatbot greeting message"""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = (
                        "Hi! Let\'s chat about movies! I\'ll help you pick a \n"
                        "movie based on your likes and dislikes. Let's get started! \n"
                        "Please tell me about a movie you've seen recently."
                        )

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return greeting_message

    def goodbye(self):
      """chatbot goodbye message"""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = 'Thanks for chatting with me!' if not self.gaveRecommendation else 'Hope I was helpful! Have fun!'

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return goodbye_message


    #############################################################################
    # 2. Modules 2 and 3: extraction and transformation                         #
    #############################################################################

    def process(self, inputStr):
      """Takes the input string from the REPL and call delegated functions
      that
        1) extract the relevant information and
        2) transform the information into a response to the user
      """
      #############################################################################
      # TODO: Implement the extraction and transformation in this method, possibly#
      # calling other functions. Although modular code is not graded, it is       #
      # highly recommended                                                        #
      #############################################################################
      if self.is_turbo == True:
        response = 'processed %s in creative mode!!' % inputStr
        return response
      else:
          return self.processSimple(inputStr)

    def processSimple(self, inputStr):
        count = inputStr.count('\"')
        if count == 0:
            no_movie = self.errors['NO_MOVIES']
            return no_movie[randint(0, len(no_movie) - 1)]
        if count != 2:
            no_quotes_list = self.errors['WRONG_QUOTES']
            return no_quotes_list[randint(0, len(no_quotes_list) - 1)]

        pattern = '\"(.*?)\"'
        matched_pattern = re.findall(pattern, inputStr)
        movie = matched_pattern[0]
        splitName = movie.split(' ')
        alternate = ''
        if splitName[0] in self.articles:
            splitName.pop(0)
            alternate = ' '.join(splitName)

        if movie in self.titles_map.keys() or alternate in self.titles_map.keys():
            key = movie if movie in self.titles_map.keys() else alternate
            index = self.titles_map[key][1]
            print index
            if self.classifySentiment(inputStr):
                self.likes_vec[index] = 1
                print 'classified good'
            else:
                self.likes_vec[index] = -1
                print 'classified bad'
            self.datapoints += 1

        if self.datapoints > self.INFO_THRESHOLD:
            recommended = self.recommend(self.likes_vec)
            top_choice = recommended[0]
            self.gaveRecommendation = True
            result += generateRecommendationString(top_choice)

        result += 'processed'
        return result

    def classifySentiment(self, inputStr):
        #TODO: Make this more robust. Right now, it's very rudimentary.
        #Just adds, subtracts sentiment words.
        split = inputStr.split(' ')
        score = 0
        for i in range(len(split)):
            split[i] = self.stemmer.stem(split[i])

        return True

    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################
    def generateRecommendationString(self, choice):
        #TODO: Make this more robust, randomly generate possibilities, etc.
        return 'You might like ' + choice

    def readInFile(self, filename):
        content = []
        with open(filename) as f:
            content = f.readlines()

        res = {}
        curr_key = ''
        for sent in content:
            if '***' in sent:
                new = sent.replace('*', '')
                new = new.rstrip()
                res[new] = []
                curr_key = new
            else:
                res[curr_key].append(sent.rstrip())

        return res

    def read_data(self):
      """Reads the ratings matrix from file"""
      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, self.ratings = ratings()
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      self.sentiment = dict(reader)

    def processTitles(self, titles_list):
        res = {}
        for i in range(len(titles_list)):
             title = titles_list[i][0]
             res[title] = [titles_list[i][1], i]
        return res

    def binarize(self):
      """Modifies the ratings matrix to make all of the ratings binary"""
      # QUESTION: Should we consider normalizing by subtracting the user's avg rating before?
      # Seems like too much unnecessary work?
      res = np.nonzero(self.ratings)
      for val in range(0, len(res[0])):
        row = res[0][val]
        col = res[1][val]
        score = self.ratings[row][col]
        if val == 0: continue
        if val < 3.0: self.ratings[row][col] = -1
        else: self.ratings[row][col] = 1

    def distance(self, u, v):
      """Calculates a given distance function between vectors u and v"""
      # TODO: Implement the distance function between vectors u and v]
      # Note: you can also think of this as computing a similarity measure
      # Right now, implements cosine similarity.
      return float(np.dot(u, v))/(np.norm(u)*np.norm(v))


    def recommend(self, u):
      """Generates a list of movies based on the input vector u using
      collaborative filtering"""
      # TODO: Implement a recommendation function that takes a user vector u
      # and outputs a list of movies recommended by the chatbot




    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, input):
      """Returns debug information as a string for the input string from the REPL"""
      # Pass the debug information that you may think is important for your
      # evaluators
      debug_info = 'debug info'
      return debug_info


    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
      return """
      Your task is to implement the chatbot as detailed in the PA6 instructions.
      Remember: in the starter mode, movie names will come in quotation marks and
      expressions of sentiment will be simple!
      Write here the description for your own chatbot!
      """


    #############################################################################
    # Auxiliary methods for the chatbot.                                        #
    #                                                                           #
    # DO NOT CHANGE THE CODE BELOW!                                             #
    #                                                                           #
    #############################################################################

    def bot_name(self):
      return self.name


if __name__ == '__main__':
    Chatbot()
