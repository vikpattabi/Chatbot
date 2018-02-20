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
      #Initialize relevant classes
      self.stemmer = PorterStemmer()
      self.sentiment = {}
      self.read_data()
      #User data
      self.response_indexes = {}
      #Read in responses
      self.responses = self.readInFile('deps/responses.txt', False)
      self.articles = ['The', 'A', 'An']
      self.negations = self.readInFile('deps/negations.txt', True)
      self.punctuation = '.,?!-;'
      #Binarize ratings matrix
      self.binarize()
      #Initialize relevant vars
      self.gaveRecommendation = False
      self.INFO_THRESHOLD = 5
      #Pre-process titles, ratings to make later work more efficient.
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
            no_movie = self.responses['NO_MOVIES']
            return no_movie[randint(0, len(no_movie) - 1)]
        if count != 2:
            no_quotes_list = self.responses['WRONG_QUOTES']
            return no_quotes_list[randint(0, len(no_quotes_list) - 1)]

        #Now we know there is a properly formatted, single movie in the input, although it may not be one
        #we have in our list.
        movie, alternate = self.extractMovieNames(inputStr)

        result = ''
        if self.titles_map.has_key(movie) or self.titles_map.has_key(alternate):
            key = movie if movie in self.titles_map.keys() else alternate
            index = self.titles_map[key][1]

            if self.response_indexes.has_key(index):
                result += self.alreadyHeardAboutMovie()
                return result
            score = self.classifySentiment(inputStr)
            if score > 0:
                self.response_indexes[index] = 1
                result += 'You liked ' + movie + '. '
            elif score < 0:
                self.response_indexes[index] = -1
                result += 'You did not like ' + movie + '. '
            else:
                result += self.respondToNoSentiment(movie)
                return result

        else:
            result += self.respondToUnseenMovie()

        if len(self.response_indexes.keys()) >= self.INFO_THRESHOLD:
            recommended = self.recommend(self.response_indexes)
            self.gaveRecommendation = True
            result += self.generateRecommendationString(recommended)

        return result

    def respondToNoSentiment(self, title):
        return "Sorry, I'm not quite sure how you feel about " + title + '.'

    def respondToUnseenMovie(self):
        return "I'm not familar with this movie."

    def alreadyHeardAboutMovie(self):
        #TODO: Make this more robust, randomly generate responses.
        return "You already told me about that movie."

    def classifySentiment(self, inputStr):
        #Right now, very rudimentary - use NB?
        #Seems a bit janky??
        score = 0
        split = inputStr.split(' ')
        negating = False
        for i in range(len(split)):
            word = split[i]
            word = self.stemmer.stem(word)
            if negating: word = 'NOT_' + word
            split[i] = word
            if self.punctuation in word: negating = False
            if word in self.negations:
                negating = True

        for word in split:
            if 'NOT_' in word:
                if word[4:] in self.sentiment:
                    score += (-1 if self.sentiment[word[4:]] == 'pos' else 1)
            elif word in self.sentiment:
                score += (1 if self.sentiment[word] == 'pos' else -1)
        #QUESTION:
        #Handle cases where sentiment is 0, user is neutral?
        #Difference between neutral user and not rating?
        #TELL ME MORE IF SCORE IS 0
        return (score > 0)

    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################
    def extractMovieNames(self, inputStr):
        pattern = '\"(.*?)\"'
        matched_pattern = re.findall(pattern, inputStr)
        movie = matched_pattern[0]
        splitName = movie.split(' ')
        alternate = ''
        if splitName[0] in self.articles:
            article = splitName[0]
            splitName.pop(0)
            year = splitName[len(splitName) - 1]
            splitName.pop()
            alternate = ' '.join(splitName)
            alternate += ', ' + article + ' ' + year
            alternate = alternate.rstrip()
        return movie, alternate

    def generateRecommendationString(self, choice):
        #TODO: Make this more robust, randomly generate possibilities, etc.
        return '\nI think I have enough information. You might like ' + choice

    def readInFile(self, filename, simple):
        content = []
        with open(filename) as f:
            content = f.readlines()
        if simple:
            for i in range(len(content)):
                content[i] = content[i].rstrip()
            return content

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
        #PRE-PROCESS w/ Porter Stemmer
      """Reads the ratings matrix from file"""
      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, self.ratings = ratings()
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      temp = dict(reader)
      for key in temp.keys():
        new_key = self.stemmer.stem(key)
        self.sentiment[new_key] = temp[key]

    def processTitles(self, titles_list):
        res = {}
        for i in range(len(titles_list)):
             title = titles_list[i][0].rstrip()
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
      # Right now, implements cosine similarity
      u_norm = np.linalg.norm(u)
      v_norm = np.linalg.norm(v)
      if u_norm == 0 or v_norm == 0: return 0.0
      return float(np.dot(u, v))/(np.linalg.norm(u)*np.linalg.norm(v))


    def recommend(self, indexes):
      """Generates a list of movies based on the input vector u using
      collaborative filtering"""
      # TODO: Implement a recommendation function that takes a user vector u
      # and outputs a list of movies recommended by the chatbot
      user_max = float('-inf')
      movie_to_recommend = ''
      for i in range(len(self.ratings)):
          if i in self.response_indexes.keys(): continue

          v = self.ratings[i]
          score = 0.0
          for j in self.response_indexes.keys():
              u = self.ratings[j]
              dist = self.distance(u, v)
              user_score = float(self.response_indexes[j])
              score += user_score * dist
          if score > user_max:
              user_max = score
              movie_to_recommend = self.titles[i][0]

      return movie_to_recommend


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
