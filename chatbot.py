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
import string

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
      #Read in data
      self.responses = self.readInFile('deps/responses.txt', False)
      self.articles = ['The', 'A', 'An']
      self.negations = self.readInFile('deps/negations.txt', True)
      self.punctuation = '.,?!-;'
      self.no_words = self.readInFile('deps/no_words.txt', True)
      self.yes_words = self.readInFile('deps/yes_words.txt', True)
      self.intensifiers = self.readInFile('deps/intensifiers.txt', True)
      self.strong_negative = self.readInFile('deps/strong_negative.txt', True)
      self.strong_positive = self.readInFile('deps/strong_positive.txt', True)
      #Binarize ratings matrix
      self.binarize()
      self.justGaveRec = False
      #Initialize relevant vars
      self.recommendations = []
      self.INFO_THRESHOLD = 5
      #Pre-process titles, ratings to make later work more efficient.
      self.titles_map = self.processTitles(self.titles)
      ## Remember which movies were mentioned without an explicit sentiment
      self.mentioned_movies = []
      self.justFollowedUp = False

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
      key = 'GOODBYE_GAVE_REC' if len(self.recommendations) > 0 else 'GOODBYE_NO_REC'
      options = self.responses[key]
      goodbye_message = options[randint(0, len(options) - 1)]

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
        #If we just gave a rec, maybe the user wants to hear more
        #instead of just inputting new movies immediately

        movie, test = self.isAClarification(inputStr)
        if self.justFollowedUp and test:
          last_mention = self.mentioned_movies.pop(self.mentioned_movies.index(movie))
          inputStr = '\"' + last_mention + '\" ' + inputStr
        self.justFollowedUp = False

        if self.justGaveRec:
            affirmation = self.classifyAffirmation(inputStr)
            if affirmation > 0:
                return self.outputRecommendation('')
            elif affirmation == 0:
                return self.outputConfusion() + ' ' + self.outputRecQuestion()
            else:
                self.justGaveRec = False
                return self.outputCuriosity()

        #Now we know there is a properly formatted, single movie in the input, although it may not be one
        #we have in our list.
        #Furthermore, we are not testing an affirmation.
        movie, alternate, count = self.extractMovieNames(inputStr)

        if count == 0:
            no_movie = self.responses['NO_MOVIES']
            return no_movie[randint(0, len(no_movie) - 1)]
        if count != 2:
            no_quotes_list = self.responses['WRONG_QUOTES']
            return no_quotes_list[randint(0, len(no_quotes_list) - 1)]

        #Declare res string
        result = ''


        self.checkForDisambiguation(movie, alternate)
        #Check if we've already seen the movie so we can reprompt
        index = -1
        if self.titles_map.has_key(movie) or self.titles_map.has_key(alternate):
            key = movie if movie in self.titles_map.keys() else alternate
            index = self.titles_map[key][1]
            if self.response_indexes.has_key(index):
                return self.alreadyHeardAboutMovie()
        #Filter input string to remove bias from movie names (if any)
        inputStr = re.sub('".*?"', 'MOVIE', inputStr)
        score = self.classifySentiment(inputStr)
        if score > 0:
            result += 'You liked ' + movie + '. '
            if index != -1: self.response_indexes[index] = 1
        elif score < 0:
            result += 'You did not like ' + movie + '. '
            if index != -1: self.response_indexes[index] = -1
        else:
          # If no sentiment was expressed, queue this movie up
          # and ask for future comments.
          self.mentioned_movies.append(movie)
          result += self.respondToNoSentiment(movie) + ' '
          self.justFollowedUp = True
          self.removeDuplicates()

        #Different output if the movie isn't in our repository of movies.
        if index == -1:
            result += self.respondToUnseenMovie() + ' '

        #If we have enough info now to recommend something.
        if len(self.response_indexes.keys()) >= self.INFO_THRESHOLD:
            return self.outputRecommendation(result)
        else: #Else, ask for more info about other movies
            if not self.justFollowedUp:
                result += self.outputCuriosity() + " "
                if len(self.mentioned_movies) > 0 :
                  result += self.outputFollowUp()
                  self.justFollowedUp = True

        return result

    def isAClarification(self, input):
      if len(self.mentioned_movies) == 0: return '', False
      movie, alternate, count = self.extractMovieNames(input)
      if count == 2:
          for name in self.mentioned_movies:
              if name == movie or name == alternate:
                  return movie if name == movie else alternate, True
          return '', False

      checkNext = False
      input = input.translate(None, string.punctuation)
      for word in input.split():
        if checkNext and word in ['movie', 'Movie']: return self.mentioned_movies[-1], True
          #You can't split like this?
        if word in ["It", 'it']: return self.mentioned_movies[-1], True
        if word in ['that', 'That']:
            checkNext = True
      return '', False

    def outputFollowUp(self):
      options = self.responses['FOLLOWUP']
      selected = options[randint(0, len(options) - 1)]
      return selected.replace('REPL', '"' + self.mentioned_movies[-1] + '"')

    def removeDuplicates(self):
    # Removes duplicates from self.mentioned_movies while preserving the order
      mentioned_set = set()
      no_duplicates = []
      for m in self.mentioned_movies:
        if m not in mentioned_set:
          mentioned_set.add(m)
          no_duplicates.append(m)
      self.mentioned_movies = no_duplicates

    def outputConfusion(self):
        options = self.responses['CONFUSION']
        selected = options[randint(0, len(options) - 1)]
        return selected

    def outputCuriosity(self):
        options = self.responses['CURIOSITY']
        selected = options[randint(0, len(options) - 1)]
        return selected

    def outputRecommendation(self, currString):
        recommended = self.recommend(self.response_indexes)
        self.recommendations.append(recommended)
        currString += ' ' + self.generateRecommendationString(recommended)
        self.justGaveRec = True
        return currString

    def checkForDisambiguation(movie, alternate):
        

    def classifyAffirmation(self, inputStr):
        inputStr = inputStr.translate(None, string.punctuation)
        inputStr = (inputStr.lower()).split(' ')
        count = 0.0
        for word in inputStr:
            if word in self.yes_words:
                count += 1
            elif word in self.no_words: count -= 1
        return count

    def respondToNoSentiment(self, title):
        options = self.responses['NO_SENTIMENT']
        selected = options[randint(0, len(options) - 1)]
        return selected.replace('REPL', '"' + title + '"')

    def respondToUnseenMovie(self):
        options = self.responses['UNSEEN_MOVIE']
        selected = options[randint(0, len(options) - 1)]
        return selected

    def alreadyHeardAboutMovie(self):
        options = self.responses['ALREADY_SAW_MOVIE']
        selected = options[randint(0, len(options) - 1)]
        return selected

    def classifySentiment(self, inputStr):
        #Right now, very rudimentary - use NB?
        #Seems a bit janky??
        score = 0
        split = inputStr.split(' ')
        negating = False
        for i in range(len(split)):
            word = split[i]
            word = self.stemmer.stem(word)
            if negating: word = 'NOT' + word
            split[i] = word
            if self.punctuation in word: negating = False
            if word in self.negations:
                negating = True

        for word in split:
            word = word.translate(None, string.punctuation)
            if 'NOT' in word:
                sliced = word[3:]
                if self.sentiment.has_key(self.stemmer.stem(sliced)) or self.sentiment.has_key(sliced):
                    score += (-1 if self.sentiment[word[3:]] == 'pos' else 1)
            elif self.sentiment.has_key(self.stemmer.stem(word)) or self.sentiment.has_key(word):
                score += (1 if self.sentiment[word] == 'pos' else -1)

        #QUESTION:
        #TELL ME MORE IF SCORE IS 0
        #NEEDS IMPROVEMENT?
        return score

    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################
    def extractMovieNames(self, inputStr):
        #Assuming we haven't just given a rec, and we're reading in movies.
        count = inputStr.count('\"')
        if count != 2: return '', '', count

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
        return movie, alternate, count

    def generateRecommendationString(self, choice):
        #TODO: Check if choice has an article in it, reformat accordingly.
        split = choice.split(' ')
        if split[-2] in self.articles:
            article = split[-2]
            if len(split) >= 3: split[-3] = split[-3].replace(',', '')
            split.pop(-2)
            choice = article + ' ' + ' '.join(split)

        options = self.responses['READY_TO_REC']
        selected = options[randint(0, len(options) - 1)]
        res = selected + ' '

        res += 'You might like \"' + choice + '\". '
        res += self.outputRecQuestion()

        return res

    def outputRecQuestion(self):
        options = self.responses['REC_QUESTION']
        selected = options[randint(0, len(options) - 1)]
        return selected

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
      self.sentiment = dict(reader)

      #QUESTION: SHOULD WE STEM?
      """
      for key in temp.keys():
        new_key = self.stemmer.stem(key)
        self.sentiment[new_key] = temp[key]
      """

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
        if val < 2.5: self.ratings[row][col] = -1
        elif val > 2.5: self.ratings[row][col] = 1

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
          if self.titles[i][0] in self.recommendations: continue

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
      Steve: a simple chatbot for movie recommendations.
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
