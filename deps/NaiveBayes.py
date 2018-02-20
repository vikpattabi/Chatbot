import sys
import getopt
import os
import math
import operator
import string

class NaiveBayes:
  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """NaiveBayes initialization"""
    self.negationList = set(self.readFile('deps/negations.txt'))
    self.punctuation = set('.!,?;')

    #Initializing my data structures:
    self.numPos = 0
    self.numNeg = 0
    self.numNegWords = 0
    self.numPosWords = 0
    self.wordCountsPos = {}
    self.wordCountsNeg = {}
    self.V = 0

  #############################################################################
  # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
  # Boolean (Binarized) features.
  # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
  # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
  # that relies on feature counts.
  #
  # If the BEST_MODEL flag is true, include your new features and/or heuristics that
  # you believe would be best performing on train and test sets.
  #
  # If any one of the FILTER_STOP_WORDS, BOOLEAN_NB and BEST_MODEL flags is on, the
  # other two are meant to be off. That said, if you want to include stopword removal
  # or binarization in your best model, write the code accordingly
  #
  # Hint: Use filterStopWords(words) defined below
  def classify(self, words):
    posProb = 0.0
    negProb = 0.0
    probPosKlass = float(self.numPos) / (self.numPos + self.numNeg)
    probNegKlass = float(self.numNeg) / (self.numPos + self.numNeg)

    #We use this to weight words at the end more when doing our best model
    words = self.preprocessAddNot(words)
    words = set(words)

    for w in words:
        if (not self.wordCountsPos.has_key(w)) and (not self.wordCountsNeg.has_key(w)):
            #The key isn't in our training set, so we skip.
            continue
        posCount = self.wordCountsPos[w] if self.wordCountsPos.has_key(w) else 0#(1.5*self.wordCountsPos[w] if self.wordCountsPos.has_key(w) else 0)
        negCount = self.wordCountsNeg[w] if self.wordCountsNeg.has_key(w) else 0#(1.5*self.wordCountsNeg[w] if self.wordCountsNeg.has_key(w) else 0)
        posProb += math.log((posCount + 1.0)/(self.numPosWords + self.V))
        negProb += math.log((negCount + 1.0)/(self.numNegWords + self.V))

    posProb += math.log(probPosKlass)
    negProb += math.log(probNegKlass)
    return (True if posProb > negProb else False)

  def preprocessAddNot(self, words):
      addingNot = False
      i = 0
      for w in words:
          if w in self.punctuation:
              addingNot = False
              i += 1
              continue
          if addingNot:
              words[i] = 'NOT_' + w
          if w in self.negationList:
              addingNot = True
          i += 1
      return words

  #Adds an example to the classifier.
  def addExample(self, klass, words):
    if klass == 'pos':
        self.numPos +=1
    else: self.numNeg += 1

    words = self.preprocessAddNot(words)
    words = set(words)

    for w in words:
        if klass == 'pos':
            if self.wordCountsPos.has_key(w): self.wordCountsPos[w] += 1
            else:
                if not self.wordCountsNeg.has_key(w): self.V += 1
                self.wordCountsPos[w] = 1
            self.numPosWords += 1
        else:
            if self.wordCountsNeg.has_key(w): self.wordCountsNeg[w] += 1
            else:
                if not self.wordCountsPos.has_key(w): self.V += 1
                self.wordCountsNeg[w] = 1
            self.numNegWords += 1

  #############################################################################


  def readFile(self, fileName):
    #Reads in a "list" file
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line.rstrip())
    f.close()
    return contents
