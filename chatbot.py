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
      self.articles = ['the', 'a', 'an']
      self.negations = self.readInFile('deps/negations.txt', True)
      self.punctuation = '.,?!-;'
      self.no_words = self.readInFile('deps/no_words.txt', True)
      self.yes_words = self.readInFile('deps/yes_words.txt', True)
      self.findpatterns = [
      #patterns for finding movies without quotes
      '\"(.*?)\"',
      'movie.*?(?:was|is|start(?:ed|s)|end(?:ed|s)) (.*)',
      '(?:I (?:think|thought|feel|felt) | watching )?(.*?) (?:was|is|start(?:ed|s)|end(?:ed|s)) .*?',
      'I .*? watching (.*)',
      'I .*?(?:watch|enjoy|hat|(?:dis)?lik|lov)ed (.*)'
       ]

      #Read in fine-sentiment data
      self.intensifiers = self.readInFile('deps/intensifiers.txt', True)
      self.strong_negative = self.readInFile('deps/strong_negative.txt', True)
      self.strong_negative = [self.stemmer.stem(word) for word in self.strong_negative]
      self.strong_positive = self.readInFile('deps/strong_positive.txt', True)
      self.strong_positive = [self.stemmer.stem(word) for word in self.strong_positive]

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
      self.checkingDisamb = False

      self.prevEmotion = 0
      self.emotionWords = self.readInEmotions()

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

      if 'REPL' in goodbye_message:
          movie = self.recommendations[randint(0, len(self.recommendations) - 1)]
          movie = re.sub('\(\d\d\d\d\)', '', movie)
          movie = self.fixArticle(movie)
          goodbye_message = re.sub('REPL', '\"' + movie + '\"', goodbye_message)

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return goodbye_message


    #############################################################################
    # 2. Modules 2 and 3: extraction and transformation                         #
    #############################################################################
    def process(self, inputStr):
        if self.is_turbo:
            return self.processTurbo(inputStr)
        else:
            return self.processSimple(inputStr)

    def processTurbo(self, inputStr):
        if self.checkingDisamb: return self.respondToDisamb(inputStr)

        #If we just gave a rec, maybe the user wants to hear more
        #instead of just inputting new movies immediately
        movie = ''
        movie, test = self.isAClarification(inputStr)
        if self.justFollowedUp and test:
          last_mention = self.mentioned_movies.pop(self.mentioned_movies.index(movie))
          if movie not in inputStr: inputStr = '\"' + last_mention + '\" ' + inputStr

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
        #TODO: implement extractMovieNamesCreative HERE
        if not self.justFollowedUp or movie == '':
            movie, alternate = self.extractMovieNamesCreative(inputStr)
        self.justFollowedUp = False


        if movie == None:
            #Check for emotions from the user
            emotions = self.checkEmotions(inputStr)
            response, emotion_felt = self.respondToEmotion(emotions)
            if emotion_felt: return response
            sentiment = self.classifySentiment(inputStr)
            #If no emotion felt on the scale, maybe check for sentiment?
            if sentiment < 0:
                options = self.responses['SADNESS']
                return options[randint(0, len(options) - 1)] + ' ' + self.outputCuriosity()
            elif sentiment > 0:
                options = self.responses['JOY']
                return options[randint(0, len(options) - 1)] + ' ' + self.outputCuriosity()
            #Otherwise, there's nothing
            no_movie = self.responses['NO_MOVIES']
            return no_movie[randint(0, len(no_movie) - 1)]


        #Declare res string
        result = ''

        #Check if we've already seen the movie so we can reprompt
        index = -1
        if self.titles_map.has_key(movie) or self.titles_map.has_key(alternate):
            key = movie if movie in self.titles_map.keys() else alternate
            index = self.titles_map[key][1]
            if self.response_indexes.has_key(index):
                return self.alreadyHeardAboutMovie()

        inputStr = re.sub(movie, 'MOVIE', inputStr)

        if index == -1:
            matches = self.checkForDisambiguation(movie, alternate)
            self.disambMatches = matches
            if len(matches) > 1:
                self.checkingDisamb = True
            elif len(matches) == 1:
                movie = matches[0]

        #In case we have a disambiguation that needs fixing?
        if self.titles_map.has_key(movie):
            index = self.titles_map[movie][1]
            if self.response_indexes.has_key(index):
                return self.alreadyHeardAboutMovie()

        movie = self.fixArticle(movie)
        #Filter input string to remove bias from movie names (if any)
        score = self.classifySentiment(inputStr)
        if self.checkingDisamb:
            self.storedScore = score
            return self.specifyDisambiguation(len(matches), movie)

        if score != 0 or (self.prevEmotion != 0 and ('also' in inputStr or 'but' in inputStr)):
          if score == 0 and 'but' in inputStr:
              self.prevEmotion *= -1
          if score == 0: score = self.prevEmotion
          if score == 1: result += 'You liked \"' + movie + '\". '
          elif score > 1 and score < 4: result += 'You really liked \"' + movie + '\". '
          elif score >= 4: result += 'You really loved \"' + movie + '\". Awesome! '
          elif score == -1: result += 'You did not like \"' + movie + '\". '
          elif score < -1 and score > -4: result += 'You really did not like \"' + movie + '\". '
          elif score <= -4: result += 'You really hated \"' + movie + '\". I will speak no more of that movie!'
          if index != -1:
              if  score < 0:
                  self.response_indexes[index] = -1
              else:
                  self.response_indexes[index] = 1
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

        self.prevEmotion = score

        return result

    def processSimple(self, inputStr):
        #If we just gave a rec, maybe the user wants to hear more
        #instead of just inputting new movies immediately
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
            #Check for emotions from the user
            #emotions = self.checkEmotions(inputStr)
            #response, emotion_felt = self.respondToEmotion(emotions)
            #if emotion_felt: return response
            no_movie = self.responses['NO_MOVIES']
            return no_movie[randint(0, len(no_movie) - 1)]
        if count != 2:
            no_quotes_list = self.responses['WRONG_QUOTES']
            return no_quotes_list[randint(0, len(no_quotes_list) - 1)]

        #Declare res string
        result = ''

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
            result += 'You liked \"' + movie + '\". '
            if index != -1: self.response_indexes[index] = 1
        elif score < 0:
            result += 'You did not like \"' + movie + '\". '
            if index != -1: self.response_indexes[index] = -1
        else:
          # If no sentiment was expressed, queue this movie up
          # and ask for future comments.
          #self.mentioned_movies.append(movie)
          result += self.respondToNoSentiment(movie) + ' '
          #self.justFollowedUp = True
          #self.removeDuplicates()

        #Different output if the movie isn't in our repository of movies.
        if index == -1:
            result += self.respondToUnseenMovie() + ' '

        #If we have enough info now to recommend something.
        if len(self.response_indexes.keys()) >= self.INFO_THRESHOLD:
            return self.outputRecommendation(result)
        else: #Else, ask for more info about other movies
            #if not self.justFollowedUp:
                result += self.outputCuriosity()
                #if len(self.mentioned_movies) > 0 :
                  #result += self.outputFollowUp()
                  #self.justFollowedUp = True

        return result

    def respondToEmotion(self, emotions):
        emotion = ''
        max_count = 0
        for word in emotions.keys():
            if emotions[word] > max_count:
                max_count = emotions[word]
                emotion = word
        key = ''
        if emotion == 'ang':
            key = 'ANGER'
        elif emotion == 'ant':
            key = 'ANTICIPATION'
        elif emotion == 'j':
            key = 'JOY'
        elif emotion == 't':
            key = 'TRUST'
        elif emotion == 'f':
            key = 'FEAR'
        elif emotion == 'su':
            key = 'SURPRISE'
        elif emotion == 'sa':
            key = 'SADNESS'
        elif emotion == 'd':
            key = 'DISGUST'

        if key == '':
            return key, False
        options = self.responses[key]
        return options[randint(0, len(options) - 1)] + ' ' + self.outputCuriosity(), True


    def respondToDisamb(self, inputStr):
        result = ''
        #Clarify how this will be tested?
        pattern = '\"(.*?)\"'
        matched_pattern = re.findall(pattern, inputStr)
        movie = ''
        if len(matched_pattern) == 0:
            for match in self.disambMatches:
                if len(re.findall('\W' + inputStr.lower() + '\W', match.lower())) > 0:
                    movie = match
                    break
        else:
            movie = matched_pattern[0]

        if self.titles_map.has_key(movie):
            self.checkingDisamb = False
            index = self.titles_map[movie][1]
            fixed = self.fixArticle(movie)
            if self.storedScore > 0:
                result += 'You liked \"' + fixed + '\".'
                self.response_indexes[index] = 1
            elif self.storedScore < 0:
                result += 'You did not like ' + fixed + '.'
                self.response_indexes[index] = -1
            else:
                self.mentioned_movies.append(fixed)
                result += self.respondToNoSentiment(fixed) + ' '
                self.justFollowedUp = True
                self.removeDuplicates()
                return result
        else:
            return self.outputConfusion()

        if len(self.response_indexes.keys()) >= self.INFO_THRESHOLD:
            return result + self.outputRecommendation('')

        result += ' ' + self.outputCuriosity()
        return result

    def checkEmotions(self, inputStr):
        inputStr = inputStr.translate(None, string.punctuation)
        res = {}
        split = (inputStr.strip()).split(' ')
        for word in split:
            word = word.replace(',', '')
            if self.emotionWords.has_key(self.stemmer.stem(word.lower())):
                vals = self.emotionWords[self.stemmer.stem(word.lower())]
                for v in vals:
                    if res.has_key(v):
                        res[v]  = res[v] + 1
                    else:
                        res[v] = 1
        return res

    def fixArticle(self, movie):
        res = ''
        article = ''
        for split in movie.split(' '):
            if split.lower() in self.articles:
                article = split
                continue
            res += split.replace(',', '') + ' '
        if article != '':
            res = article + ' ' + res
        return res.strip()

    def specifyDisambiguation(self, num, input):
        options = self.responses['DISAMB']
        selected = options[randint(0, len(options) - 1)]
        selected = selected.replace('REP2', '\"' + input + '\"')
        return selected.replace('REPL', str(num))

    def isAClarification(self, input):
      if len(self.mentioned_movies) == 0: return '', False
      movie, alternate = self.extractMovieNamesCreative(input)
      for name in self.mentioned_movies:
          if name == movie or name == alternate:
              return movie if name == movie else alternate, True
      return self.mentioned_movies[-1], True

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

    def checkForDisambiguation(self, movie, alternate):
        matches = []
        if alternate != '':
            for name in self.titles_map.keys():
                if movie.lower() in name.lower() or alternate.lower() in name.lower():
                    match = movie if movie in name else alternate
                    matches.append(name)
        else:
            for name in self.titles_map.keys():
                if movie.lower() in name.lower():
                    matches.append(name)

        return matches


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
        key = 'NO_SENTIMENT_SIMPLE' if not self.is_turbo else 'NO_SENTIMENT'
        options = self.responses[key]
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

    def calculateScore(self, word, prev_word, prev_prev_word):
        stemmed = self.stemmer.stem(word)
        score = 0
        if self.sentiment.has_key(word) or self.sentiment.has_key(stemmed):
            w = word if self.sentiment.has_key(word) else stemmed
            if self.sentiment[w] == 'pos': score += 1
            if self.sentiment[w] == 'neg': score += -1
        if not self.is_turbo: return score

        if word in self.strong_positive or stemmed in self.strong_positive: score += 2
        if word in self.strong_negative or stemmed in self.strong_negative: score += -2

        if prev_word != "" and prev_word in self.intensifiers:
            score *= 2
        if prev_prev_word != "" and prev_prev_word in self.intensifiers:
            score *= 2
        return score

    def classifySentiment(self, inputStr):
        score = 0
        split = inputStr.split(' ')
        negating = False
        for i in range(len(split)):
            word = split[i]
            #word = self.stemmer.stem(word)
            if negating: word = 'NOT' + word
            split[i] = word
            if self.punctuation in word: negating = False
            if word in self.negations:
                negating = True

        prev_prev_word = ""
        prev_word = ""
        for word in split:
            word = word.translate(None, string.punctuation)
            coefficient = 1
            if 'NOT' in word:
                word = word[3:]
                coefficient = -1
            score += coefficient * self.calculateScore(word, prev_word, prev_prev_word)
            prev_prev_word = prev_word
            prev_word = word

        if '!' in inputStr: score *= 2
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
            if '(' in splitName[len(splitName) - 1]:
                year = splitName[len(splitName) - 1]
                splitName.pop()
                alternate = ' '.join(splitName)
                alternate += ', ' + article + ' ' + year
                alternate = alternate.strip()
            else:
                alternate = ' '.join(splitName) + ', ' + article
        return movie, alternate, count

    def searchForPatterns(self, searchstring, pattern_strings):
        patterns = []
        for pattern_string in pattern_strings:
            patterns.append(re.compile(pattern_string, re.IGNORECASE))

        # Iterate over the lines of the file.
        line = searchstring
        matched_groups = {}
        for i, p in enumerate(patterns):
            m = p.findall(line)
            if m != []:
                matched_groups[i] = m[0]

        # Return the match corresponding to the first regular expression in the list.
        if len(matched_groups) > 0:
            return matched_groups[min(matched_groups.keys())]
        else:
            return None

    def tryFullSearch(self, inputStr):
        inputStr = inputStr.translate(None, string.punctuation)
        inputStr = ' ' + inputStr + ' '
        inputStr = inputStr.lower()
        print inputStr
        allMovies = self.titles_map.keys()
        for movie in allMovies:
            no_year = re.sub('\(\d\d\d\d\)', '', movie)
            no_year = no_year.strip()
            no_year = ' ' + no_year.lower() + ' '
            if no_year == ' saw ': no_year = ' Saw '
            if no_year in inputStr:
                return self.fixArticle(no_year.strip()), no_year.strip()
        return None, None


    def extractMovieNamesCreative(self, inputStr):
        movie, alternate = self.tryFullSearch(inputStr)
        if movie == None:
            movie = self.searchForPatterns(inputStr, self.findpatterns)
        else:
            movie = movie.strip()
            alternate = alternate.strip()
            return movie, alternate
        # print movie
        # print '----------------------------------'
        if movie == None:
            return None, None

        splitName = movie.split(' ')
        alternate = ''
        if splitName[0] in self.articles:
            article = splitName[0]
            splitName.pop(0)
            year = splitName[len(splitName) - 1]
            splitName.pop()
            alternate = ' '.join(splitName)
            alternate += ', ' + article + ' ' + year
            alternate = alternate.strip()
        return movie, alternate

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

    def readInEmotions(self):
        content = []
        with open('deps/smaller.txt') as f:
            content = f.readlines()
        res = {}
        for line in content:
            line = line.strip()
            split = line.split(' ')
            word = split[0]
            split.pop(0)
            res[word] = split
        return res


    def readInFile(self, filename, simple):
        content = []
        with open(filename) as f:
            content = f.readlines()
        if simple:
            for i in range(len(content)):
                content[i] = content[i].strip()
            return content

        res = {}
        curr_key = ''
        for sent in content:
            if '***' in sent:
                new = sent.replace('*', '')
                new = new.strip()
                res[new] = []
                curr_key = new
            else:
                res[curr_key].append(sent.strip())

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

      #QUESTION: SHOULD WE STEM?

      for key in temp.keys():
        new_key = self.stemmer.stem(key)
        self.sentiment[new_key] = temp[key]


    def processTitles(self, titles_list):
        res = {}
        for i in range(len(titles_list)):
             title = titles_list[i][0].strip()
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

      In creative mode, the following features are implemented:
      -Identifying movies without quotes or with imperfect capitalization.
      -Fine-grained sentiment extraction.
      -Disambiguating movie titles (also works for when you don't include the article or year).
      -Identifying and responding to emotions.
      -Understanding references to things said previously.
      -Responding to (some) arbitrary inputs.
      -(Hopefully) speaking very fluently.
      -Responding to common affirmations / "no-words" when asking if the user wants another recommendation.

      -Note: Creative mode and simple mode are split up and should be tested independently.
      -Note: The creative features are integrated together (and can be tested together in case that happens to be worth extra points).

      Happy grading!

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
