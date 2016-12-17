# IS620 - IS620 Final Project
# Program: IS60FinalProjectPhase8.py
# Student: Neil Acampa
# Date:   11/18/16
# Function:


#    Version 8 addes classification: neutral for ratings of 3  
#    Need to work on neutral 12/12/16-12/13/16
#    Ratings:1-2 = negative
#    Ratings:4-5 = postive
#    Rating: 3   = neutral

#    For version 7 - Separate bag of words apsect from classification

#    Since Bayesian method sometimes calculate a sentence with Negative and Positive class equal
#    This would be the same as an external rating equalling 3

#    This version will work for a book (Alice in Wonderland, Far from the Maddening crowd and Mobey Dick)
#    Choose a corpus of interest and perform classification  

#    Found book review data on 11/29/16 from 						  Complete 11/30/16
#    http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html


#    Read in AFINN, Harvard words and other sentiment work value documents		  Complete 11/19/16
#    Read in WordStat sentiment dictionary						  Not Yet
#    Read in MPQA									  Not yet

#    As a test read in any document Alice in Wonderland since review's will be in a similar format
#    Parse sentences				  				          Complete 11/19/16


#    For each predefined keyword  (Need to add a few more)
#    Get the Synset definition and name	and store in an array				  In process 11/30/16
#    Key Aspect words:
#    Science Fiction
#    Myths
#    Fantasy
#    Fairy Tale
#    Mystery
#    

#    For each sentence find the keyword(s) 				        	 			Complete 11/20/16
#    Then once found calculate

#    Then do Multiple methods (Bag of words with Synset)

#    Method 1.1 (Bag of Words)
#      Tally the total sentence score using AFINN and Harvard words			  			Complete 11/19/16

#    Method 1.2: Synset Definition and value	
#    Get synsets definition and then from the definition get Synsets pos/neg word value	  			Complete 11/19/16


#    In methods 1.1 and 1.2 we are using dictionaries of sentiment scores and Synsets pos/neg word value
#    to determine score. Evaluate against review rating	   						        Complete 12/10/16

#    For each Aspect or Keyword show the total Pos/Neg score, the average of Pos/Negative rating		Complete 12/11/16
#    show the sentence and the +/- and rating




#     Use negative word	logic										       Not Yet


#     Method 2 is a statistical 
#     2.1 Naive Bayes						  						Complete 12/03/16

#     Probability of a Class (Positive) is given by the Posterial Probability of (Class) given all Documents	Complete 12/03/16
#     So Class(positive) = Total positive sentences / All Sentences with Aspect
#     So Class(negative) = Total negative sentences / All Sentences with Aspect


#     Count total Aspect Words and Count Total Pos aspect sentence words and total neg aspect sentence words	Complete 12/03/16
#     Then for a sepecific word(s) in a sentence belonging to a class						Complete 12/03/16

#     Show Training set words in a table
#     Show word, total occurrances, total Positive, total Negative 		     				Complete 12/04/16



#     prob = class positive prob * (for each word) get the number of times specific word is positive/total positive words
#     prob neg = classs neg prob * (for each word in the sentence the number of times specific word negative/total negative words

#     We estimate the prior class probabilities from a training set using rating values				Complete 11/03/16

#     The training set can then be used to classify test documents						Complete 11/04/16
 
#     On the test set - we don't use the rating score - we just classify					Complete 11/05/16

#     Train on 80% test on 20%											Complete 12/08/16

					
#     Method 2.2 execute same Naive Bayes but use binary representaion of a sentent word			Complete 12/05/16
#     (If a word occurs more than once update the array one time only)
													
				
					
#     Read book revies - xml file -  XML is not formed correctly 						Complete 12/11/16										 Started 12/09/16
#     Read and parse manually											Complete 12/11/16

#    If sentence sentiment value is not zero								        Complete 12/09/16					  			
#    Update keyword array with the book name, title, rating and sentence




#    Posible use logistic regresion with 1/0 as target

#    Also Pass 2 using LSI AND SVD to find matches
#    for all review words add to vocabulary
#    create matrix with review words in column 1 and all other columns
#    represent frequency of words for book

#    For each aspect sentence found, use LSI and SVD and match sentence found with all
#    reviews using cosine similarty. Take the average rating of the top 10 results
#    and use as sentiment score.
#    We are now matching the aspect sentence with a low level approximation of the entire
#    book review space and then averaging the top ratings.



#    For Alice in Wonderland keywords:
#    look, fall, down, dry, queen


#    For example Key words in Restaurant
#    Service, Food, Decor, Ambiance, Atmosphere, Location, Parking
#    
#    Key words for Product: Camera
#    Ease of use, Megapixels, FPS (Framse per second), Zoom, Batteries



from __future__ import absolute_import 
from __future__ import division
import re
import os 
import math
import decimal
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import networkx as nx
import random
from urllib import urlopen
import nltk
nltk.download('gutenberg')
from nltk import word_tokenize
nltk.download('maxent_treebank_pos_tagger')
nltk.download('punkt')
nltk.download('movie_reviews')
nltk.download('stopwords')
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')
from nltk.corpus import senseval
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.corpus import sentiwordnet as swn
from xml.dom import minidom
from xml.dom.minidom import parse, parseString
from nltk import sent_tokenize


linelst=[]
lines  = ""

# Arrays to save Product, Title, Ratind and Review
prodnames	  = []
titles		  = []
ratings		  = []
reviews		  = []

allwords          = []   # Contains all words

# The arrays below are used so we can match the document word to a sentiment value
# Need to find some other arrays with sent word values
sentiment         = []   # 2-D matrix sentiment word and value
sentimentdict     = []   # Contains words from AFINN, Harvard Sentiment word  Harvard sentiment word: Positive = 1, Negative = -1
sentimentdictcnt  = []   # Contains a sentiment value (-5 to +5) for sentiment word


# These arrays are used to hold Review sentence and the overall sentence sentiment value

sentword          = []   # Holds an entire sentence with comma separted words from the Corpora
sentrating        = []
senttitle         = []

aspects            = []   # Holds aspect itself
aspectsent         = []   # Holds only those sentences which include the Aspect
aspectval          = []   # Holds the corresponding aspect sentiment value 
aspecttitle        = []   # Holds the corresponding aspect book title
aspectsynval       = []   # Holds the corresponding aspect synset value
aspectsynpos       = []   # Holds the corresponding aspect synset positive value
aspectsynneg       = []   # Holds the corresponding aspect synset negative value
aspectrating       = []   # Holds the corresponding aspect rating from the review


syns              = []   # Synset id for a specific word 
synsdef           = []   # The corresponding Synset Sentence
synsname          = []   # The corresponding Synset name (i.e. 'look.v.09')





keywords          = []   # Holds predefined Aspect Keywords for project

# Aspect Keywords for Book Reviews 
# Later when all working remove Alice and change this to keywords
keywords          = []   # Holds predefined Aspect Keywords for project

keywords.append("myth")
keywords.append("fantasy")
keywords.append("tale")
keywords.append("victorian")
keywords.append("hero")
keywords.append("folk")
keywords.append("mystery")
keywords.append("magical")
keywords.append("mythology")
keywords.append("god")


# Just for testing

keywords          = []  

keywords.append("myth")
keywords.append("hero")
keywords.append("magical")
keywords.append("god")
keywords.append("science")
keywords.append("mystery")
keywords.append("fantasy")
keywords.append("tale")
keywords.append("folk")
keywords.append("mirth")
keywords.append("goddess")


keycnt = len(keywords)


# Used in Naive Bayes

allwords          = []   # Contains all words in aspect sentences with a rating
masterdict        = []   # Contains unique words in aspect sentences with a rating
masterdictcnt     = []   # Contains count of unique words corresponding to masterdict
masterdictpos     = []   # Contains all positive words in aspect sentences with a rating
masterdictposcnt  = []   # Contains count of all positive words in aspect sentences with a rating
masterdictneg     = []   # Contains all negative words in aspect sentences with a rating
masterdictnegcnt  = []   # Contains count of all negative words in aspect sentences with a rating
masterdictntrl    = []   # Contains all neutral words in aspect sentences with a rating
masterdictntrlcnt = []   # Contains count of all neutral words in aspect sentences with a rating
uniquewords       = []   # Contains unique words in the first dimension and the count in the second dim
                     

# Used in Boolean Multinimial Naive Bayes

allwords          = []   # Contains all words in aspect sentences with a rating
masterdict        = []   # Contains unique words in aspect sentences with a rating
masterdictcnt     = []   # Contains count of unique words corresponding to masterdict
masterdictpos     = []   # Contains all positive words in aspect sentences with a rating
masterdictposcnt  = []   # Contains count of all positive words in aspect sentences with a rating
masterdictneg     = []   # Contains all negative words in aspect sentences with a rating
masterdictnegcnt  = []   # Contains count of all negative words in aspect sentences with a rating

uniquewords       = []   # Contains unique words in the first dimension and the count in the second dim



# Table Elements

fheadings      = [] 



rejectchars = [',','.','?','<','>','!','"','-','%','&','#','(',')','*',';'];
rejectcnt = len(rejectchars);


def remove_characters(word):
  """Replace special characters in the word"""
  
  for i in range(rejectcnt):
    rchar = rejectchars[i]
    if rchar in word:
      word = word.replace(rchar,"")

  return word


def remove_symbols(word):
  """Replace symbols in the word"""
  w = len(word)
  word = (ord(c) for c in word) 
  word = map(lambda x:x if x<123 or x>255 else " ", word)
  newword=""
  for c in range(w):
    if word[c] <> " ":
      newword += chr(word[c]);
  
  return newword

def find_phrase(phrase, syns):
  """Find and return index of Synset in sentence"""

  masterlen = len(syns)
  find=0
  temp="x"
  try:
   temp = syns.index(phrase);
   return temp
  except ValueError:
   return temp

def find_word2(word, masterdict):
  """Find and return index of word in dictionary"""

  masterlen = len(masterdict)
  find=0
  temp="x"
  try:
   temp = masterdict.index(word);
   return temp
  except ValueError:
   return temp


def find_word1(word, sentence):
  """Find and return index of word in sentiment dictionary"""

  masterlen = len(sentence)
  find=0
  temp="x"
  try:
   temp = sentence.index(word);
   return temp
  except ValueError:
   return temp


def find_word(word, sentimentdict):
  """Find and return index of word in sentiment dictionary"""

  masterlen = len(sentimentdict)
  find=0
  temp="x"
  try:
   temp = sentimentdict.index(word);
   return temp
  except ValueError:
   return temp


def evaluate_sentences(sentence):
  """Attempt to assign a sentiment value to each word in the sentence"""
  """Return overall sentence sentiment value"""
 

  totalsentimentvalue = 0
  sentimentval        = 0
  temp = sentence.split(",");
  l = len(temp)
  for x in range(l):
     word = remove_characters(temp[x])
     word = remove_symbols(word)
     word = word.lower()
     word = word.replace(" ","")
     findx = find_word(word, sentimentdict)
     sentimentval = 0
     if (findx != "x"):
       sentimentval = int(sentimentdictcnt[findx]) + 0
       totalsentimentvalue = totalsentimentvalue + sentimentval
   
   
  return totalsentimentvalue 



def document_features(sentence, keyword):
  """ For a given word and sentence                                                    """
  """ The Lesk algorithm returns a Synset with the highest number of overlapping words """
  """ between the context sentence and different definitions from each Synset.         """

  features = {}
  synsent  = ""
  phrase   = lesk(sentence,keyword)
  findx = find_phrase(phrase, syns)
  if (findx != "x"):
    synsent = synsdef[findx]
    synsent = synsent.encode('ascii')
   
 
  features['(%s : %s)' % (sentence, phrase)]  = synsent
  
  return features


def document_features_sentiment(document):
  docwords = set(document)
  features = {}
  totalsentimentvalue = 0
  sl = len(sentimentdict)
  for word in wordfeaturesSent:
    tword = word.encode('ascii')
    findx = find_word(tword, sentimentdict)
    sentimentval = 0
    if (findx != "x"):
      sentimentval = int(sentimentdictcnt[findx]) + 0
      totalsentimentvalue = totalsentimentvalue + sentimentval
    
    
    
    features['contains(%s SV:%s)' % (word, sentimentval)] = (word in docwords)

 
  return features


# Read in words with a sentiment value from AFINN and Harvard Inquire basic
print("Reading in AFINN and Harvward Inquire Basic sentiment words")
print
filepath=""
temp    =""
tokens  = ""
valid   = 0
p       = 1
cwd = os.getcwd()
corpus     = "AFINN"
fullcorpus = "AFINN-111.txt"
currfilepath = str(cwd) + "\AFINN-111.txt"
print currfilepath
print ("Enter the Full File Path including the File")
print ("or Press return to use current File Path %s") % (currfilepath)
filepath = raw_input("Please enter the File Path now ")
valid = 0
if filepath == "":
   filepath = currfilepath

 
try:
       f = open(filepath,"r")
       try:
         valid=1
         x =0
         j=0
         for lines in f:
           lines = lines.rstrip()
           temp = lines.split("\t");
           sentimentdict.append(temp[0])
           sentimentdictcnt.append(temp[1])
                            
       finally:
            f.close()
         
except IOError:
       print ("File not Found - Program aborting")

if not(valid):
     exit()



cwd = os.getcwd()
corpus     = "inquirebasic.txt"
fullcorpus = "inquirebasic.txt"
currfilepath = str(cwd) + "\inquirebasic.csv"
print currfilepath
print ("Enter the Full File Path including the File")
print ("or Press return to use current File Path %s") % (currfilepath)
filepath = raw_input("Please enter the File Path now ")
valid = 0
if filepath == "":
   filepath = currfilepath

 
try:
       f = open(filepath,"r")
       try:
         valid=1
         x =0
         j=0
         for lines in f:
           lines = lines.rstrip()
           temp = lines.split(",");
           word = temp[0].lower()
           sentimentdict.append(word)
           if (temp[1] != ""):
             # give positive words + 1
             sentimentdictcnt.append(1)
           else:
             # give negative words -1 
             if (temp[2] != ""):
               sentimentdictcnt.append(-1)
             else:
               sentimentdictcnt.append(0)
                            
       finally:
            f.close()
         
except IOError:
       print ("File not Found - Program aborting")

if not(valid):
     exit()
 
# Store all sentiment words and values in sentiment array
sentimentcnt = len(sentimentdict)
for i in range(sentimentcnt):
  sentiment.append([sentimentdict[i], sentimentdictcnt[i]])


sl = len(sentimentdict)

print
sentences = ""
sentword  = []
print
print("Start to read Book Review Data: bookreviews.xml")
print
#print("url:   http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html")
print
corpus     = "Book Reviews"
fullcorpus = "Book Reviews"
cwd = os.getcwd()
#currfilepath = str(cwd) + "\\testa.xml"
## Test to see if can parse
currfilepath = str(cwd) + "\\bookreviews2.txt"
print
print(currfilepath)
print ("Enter the Full File Path including the File")
print ("or Press return to use current File Path %s") % (currfilepath)
filepath = raw_input("Please enter the File Path now ")
valid = 0
if filepath == "":
   filepath = "c:\\anaconda2\\bookreviews2.txt"
   print(filepath)
 
try:
       f = open(filepath,"r")
       try:
         valid=1
         x =0
         j=0
         upddate=0
         pcnt = 0
         tcnt = 0
         rcnt = 0
         rvcnt= 0
         cnt  = 0
         for lines in f:
           update    = 0
           lines     = lines.rstrip()
           cnt=cnt+1
           
           if (lines.find('<product_name>') !=-1): 
               pcnt=cnt+1

           if (lines.find('<rating>') !=-1):
               rcnt = cnt+1

           if (lines.find('<title>') != -1):
               tcnt= cnt+1
           
           if (lines.find('<review_text>') != -1):
               rvcnt = cnt+1

           if (cnt == pcnt):  
               prodnames.append(lines)
              
           if (cnt == rcnt):
              ratings.append(lines)
    
           if (cnt == tcnt):
              titles.append(lines)
              
           if (cnt == rvcnt):
              reviews.append(lines)
       
               
        
                                  
                     
        
                                  
       finally:
            f.close()
         
except IOError:
       print ("File not Found - Program aborting")

if not(valid):
     exit()



cnt=0
prodnames = np.array(prodnames)
ratings   = np.array(ratings)
reviews   = np.array(reviews)
titles    = np.array(titles)

bl = len(prodnames)-1
print
print("Total Number of Reviews Read %i") % (bl)

cnt=0

# Now do the Aspect search using the xml file
print
print("Finding Key Aspects in each Review - Please wait...")
print 
for i in range(bl):
 title   = prodnames[i]
 review  = reviews[i]
 rating  = float(ratings[i])
 rating = int(rating)
 sent = sent_tokenize(review)
 sentl = len(sent)
 for y in range(sentl):
  temp = sent[y].split(" ");
  l    = len(temp)
  senttemp = ""
  findword =0
  for x in range(l):
    word = remove_characters(temp[x])
    word = remove_symbols(word)
    word = word.lower()
    word = word.replace(" ","")
    if (word != ''):
      for j in range(keycnt):
       if (word == keywords[j]):
            findword=1
            break

      if senttemp == "":
          senttemp = word
      else:
          senttemp = senttemp + "," + word
          
           
  if (findword == 1):
     cnt+=1
     sentword.append(senttemp)
     senttitle.append(title)
     sentrating.append(rating)
     
     
  findword = 0        
           


# Now have Aspect sentences, the entire review rating and the review title
doclen   = len(sentword)
print
print("Aspect Sentences found: %s") % (doclen)

           

# Store Synset defintions for Aspect Keyword 
print("For Each Aspect get SynSet Definition")
print
for i in range(keycnt):
  keyword = keywords[i]
  print
  print("Synsets for Keyword: %s") % (keyword)
  for ss in wn.synsets(keyword):
    syns.append(ss)
    synsdef.append(ss.definition())
    temp = ss.name()
    temp = temp.encode('ascii')
    synsname.append(temp)
    print("%s\t%s") % (ss,ss.definition())


# Skip for now do bayesian
print
# Once ok move update MD here
# Update master dictionary of aspect sentences with values
print("Training on Aspect Sentences")
print("Update Master Dictionary with Aspect Sentence words")

train      = int(doclen*.80)
teststart  = train+1

print



print("Train on fist 80 percent of Aspect Sentences: %i") % (train)
print("Test on last 20 percent of Aspect Sentence:  %i to %i") % (teststart, doclen)

print
testpos  = 0
testntrl  = 0
testneg   = 0
testtotal = 0
for i in range(train):
  sentence = sentword[i]
  temp = sentence.split(",");
  l = len(temp)
  for j in range(l):
    word = remove_characters(temp[j])
    word = remove_symbols(word)
    word = word.lower()
    word = word.replace(" ","")
    if (word != ''):
      testtotal = testtotal + 1
      findx = find_word2(word, masterdict)
      if (findx == "x"):
        masterdict.append(word)
        masterdictcnt.append(1)
      else:
        masterdictcnt[findx]+=1
      
      
      if (sentrating[i] > 3):
        findx = find_word2(word, masterdictpos)
        if (findx == "x"):
          masterdictpos.append(word)
          masterdictposcnt.append(1)
        else:
          masterdictposcnt[findx]+=1
        
        testpos = testpos + 1
   
      if (sentrating[i] == 3):
        findx = find_word2(word, masterdictntrl)
        if (findx == "x"):
          masterdictntrl.append(word)
          masterdictntrlcnt.append(1)
        else:
          masterdictntrlcnt[findx]+=1
        
        testntrl = testntrl + 1

      if (sentrating[i] <= 2):
        findx = find_word2(word, masterdictneg)
        if (findx == "x"):
          masterdictneg.append(word)
          masterdictnegcnt.append(1)
        else:
          masterdictnegcnt[findx]+=1

        testneg = testneg + 1





print("Naive Bayes Statistical Method of Classification")
print
print
uniquecnt = len(masterdict)
pwordcnt=0
nwordcnt=0
ntrlwordcnt=0
print ("Statistics for Corpus:  %s") % (fullcorpus)
print
print("Total Aspcet sentences: %i") % (doclen)
print("Total Aspect words: %d from Aspect Sentences") % (testtotal)
print("Total Unique Aspect words: %d from Aspect Sentences") % (uniquecnt)
print("Total Positive Aspect words: %i Total Neutral Aspect words %i Total Negative Aspect words %i") % (testpos, testntrl, testneg)
print
print("Train on first 80 percent of Aspect Sentences %i")  % (train)
print("Test on last 20 percent of Aspect Sentence: %i to %i") % (teststart,doclen)
print
print
print
print("List of 15 unique word counts")
# Later on show all words use uniquecnt
print("Word\tTotal\tPos\tNtrl\tNeg")
for m in range(15):
 pword = masterdict[m]
 findx = find_word2(pword, masterdictpos)
 if (findx == "x"):
   pwordcnt = 0
 else:
   pwordcnt = masterdictposcnt[findx] 

 findx = find_word2(pword, masterdictntrl)
 if (findx == "x"):
   ntrlwordcnt = 0
 else:
   ntrlwordcnt = masterdictntrlcnt[findx] 

 findx = find_word2(pword, masterdictneg)
 if (findx == "x"):
   nwordcnt = 0
 else:
   nwordcnt = masterdictnegcnt[findx] 

 print("%s\t%i\t%i\t%i\t%i") % (masterdict[m], masterdictcnt[m], pwordcnt, ntrlwordcnt, nwordcnt)
 

print

print("Naive Bayes Method")
print
print("ClassNaiveBayes = argmaxP(C)*Product(i=1 to n)P(W(i)|C)")
print
print("P(C) is the Prior probabilty of a Class given the Training Set")
print("P(C) = All words in a Class over the Total number of words in the Training Set")
print
print("P(W(i)|C) = Count(W(i),C) or the count of a specific word in a Class over all words in that Class")
print

#classpospr = (testpos/doclen)
#classnegpr = (testneg/doclen)

# Revised 12/13/16 - To calculate Prior Class Probability based on the Total Class words/ All words
classpospr   = (testpos/testtotal)
classnegpr   = (testneg/testtotal)
classntrlpr  = (testntrl/testtotal)

tclasspos   = classpospr
tclassneg   = classnegpr
print
print("Prior Probability Class Positive: %.3f") % (classpospr)
print("Prior Probability Class Neutral: %.3f") %  (classntrlpr)
print("Prior Probability Class Negative: %.3f") % (classnegpr)
print


print("Training using 80 percent of Data")
print
pwordcnt    = 0
ntrlwordcnt = 0
nwordcnt    = 0
nbpos       = 0
ntrlpos     = 0
nbneg       = 0
pper        = 0
ntrlper     = 0
nper        = 0
pperarray   = []
ntrlperarray= []
nperarray   = []
tposarray   = []
tnegarray   = []
matcharray  = []
tp         = 0 # True  Positive
tn         = 0 # True  Negative
fp         = 0 # False Positive  - Classifiy as Possitive but Actually Negative or Neutral
fn         = 0 # False Negativce - Classify as Negative but Actually Positive or Neutral
tntrl      = 0 # True  Neutral    - 
fntrl      = 0 # False Neutral   - Classify as Neutral but Actually Positive or Negative
print

for i in range(teststart,doclen):
  pperarray   = [] # Array containing positive word probabilities
  ntrlperarray= [] # Array containing neutral word probabilities
  nperarray   = [] # Array containing negative word probabilities
  pper        = 0  # Probability word positive over all Positive
  nper        = 0  # Probability word negative over all Negative
  
  sentence = sentword[i]
  temp = sentence.split(",")
  l = len(temp)
  findword = 0
  for x in range(l):
    word = remove_characters(temp[x])
    word = remove_symbols(word)
    word = word.lower()
  
    # 12/14/16 - Add Laplace smoothing
    # Add 1 to the count and the length of the vocabulary to the denominator
    if (word != ''):
      findx = find_word2(word, masterdictpos)
      if (findx == "x"):
        pwordcnt = 1
      else:
        pwordcnt = masterdictposcnt[findx] 
        pwordcnt+=1

      
      pper = pwordcnt/(testpos + testtotal)
      pperarray.append(pper)
      

       
      findx = find_word2(word, masterdictntrl)
      if (findx == "x"):
        ntrlwordcnt = 1
      else:
        ntrlwordcnt = masterdictntrlcnt[findx] 
        ntrlwordcnt+=1

      if (testntrl !=0):
         ntrlper = ntrlwordcnt/(testntrl + testtotal)
      else:
         ntrlper =0

      ntrlperarray.append(ntrlper)


    
      findx = find_word2(word, masterdictneg)
      if (findx == "x"):
        nwordcnt = 1
      else:
        nwordcnt = masterdictnegcnt[findx]
        nwordcnt++1

      nper = nwordcnt/(testneg + testtotal)
      nperarray.append(nper)
      
      

  
  naivebayespos  = np.prod(np.array(pperarray))
  naivebayesntrl = np.prod(np.array(ntrlperarray))
  naivebayesneg  = np.prod(np.array(nperarray))

 

  nbpos = (classpospr  * naivebayespos)
  nbntrl= (classntrlpr * naivebayesntrl)
  nbneg = (classnegpr  * naivebayesneg) 

  
   
  
  print
  print("Sentiment Classification: Review Rating: %d : Aspect Sentence: %s") % (sentrating[i],sentword[i])
  print
  print("Naive Bayes Method")
  if (nbpos > nbneg):
    print("Classified:Positive")
    if (sentrating[i]>3):
       print("Sentence Rating Positive: Correct Classification")
       print("Aspect sentence value %i") % (sentrating[i])
       tp=tp+1     

    if (sentrating[i]==3):
       print("Sentence Rating Neutral: Incorrect Classification")
       print("Aspect setient value %i") % (sentrating[i])
       fp+=1

    if (sentrating[i]<=2):
       print("Sentence Rating Negative: Incorrect Classification")
       print("Aspect setient value %i") % (sentrating[i])
       fp+=1

  if (nbpos < nbneg):
    print("Classified: Negative")
    if (sentrating[i] <= 2):
       print("Sentence Rating Negative: Correct Classification")
       print("Aspect setient value %i") % (sentrating[i])
       tn+=1    
  
    if (sentrating[i]==3):
       print("Sentence Rating Neutral: Incorrect Classification")
       print("Aspect sentence value %i") % (sentrating[i])
       fn+=1
  
    if (sentrating[i]>3):
       print("Sentence Rating Positive: Incorrect Classification")
       print("Aspect sentence value %i") % (sentrating[i])
       fn+=1
 
  if (nbpos == nbneg):
    print("Classified: Neutral")
    if (sentrating[i] == 3):
       print("Sentence Rating Neutral: Correct Classification")
       print("Aspect setient value %i") % (sentrating[i])
       tntrl+=1 
          
    if (sentrating[i] <=2):
       print("Sentence Rating Negative: Incorrect Classification")
       print("Aspect setient value %i") % (sentrating[i])
       fntrl+=1
      
    if (sentrating[i]>3):
       print("Sentence Rating Positive: Incorrect Classification")
       print("Aspect sentence value %i") % (sentrating[i])
       fntrl+=1


print
print
print("Naive Bayes Statistical Method of Classification")
print("Evaluation")
evalstats = []
evalstats.append(tp)
evalstats.append(fp)
evalstats.append(tn)
evalstats.append(fn)
print
print("True Positive:  %i     False Positive %i")  % (tp, fp)
print("False Negative: %i     True  Negative %i") % (fn ,tn)
print("True Neutral:   %i     False Neutral  %i")  % (tntrl, fntrl)




if ((tp >0) or (fp > 0)):
    PrecisionP    = tp / (tp + fp)
else:
    PrecisionP  = 0

if ((tp > 0) or (fn > 0)):
   RecallP       = tp / (tp + fn)
else:
   RecallP = 0

if ((tntrl >0) or (fntrl > 0)):
    PrecisionNtrl   = tntrl / (tntrl + fntrl)
else:
    PrecisionNtrl  = 0

if ((tntrl > 0) or (fntrl > 0)):
   RecallNtrl       = tntrl / (tntrl + fntrl)
else:
   RecallNtrl = 0


if ((tn >0) or (fn > 0)):
    PrecisionN    = tn / (tn + fn)
else:
    PrecisionN = 0

if ((tn > 0) or (fp > 0)):
    RecallN       = tn / (tn + fp)
else:
    RecallN  = 0

Accuracy     = (tp + tn + tntrl)/(tp + tn + tntrl + fp + fn + fntrl)

totPrecision = PrecisionP + PrecisionN + PrecisionNtrl
totRecall    = RecallP    + RecallN    + RecallNtrl

if ((totPrecision > 0) or (totRecall > 0)):
  Fscore       = (2*totPrecision*totRecall)/(totPrecision + totRecall)
else:
  Fscore = 0


print("Accuracy in the context of Classification")
print("The total number of correctly predicted over all records")
print
print
print("Precision in the context of Classification:")
print("The ratio of correct predictions in a Class")
print("over all predictions for that Class: Correct and Incorrect")
print("It is a measure of exactness")
print
print("Recall in the Context of Classification:")
print("The number of correct predictions for a Class")
print("over the number of items actually belonging to that Class")
print("based on observations")
print("It is a measure of completness")
print
print("F-score balances Precision and Recall giving equal weight to both")
print
print("Observation for Book Reviews: Rating")
print("Ratings from 1-2 are negative")
print("Ratings equal to 3 are neutral")
print("Ratings from 4-5 are positive")
print
print("Expection is based on the Probability distributions")
print

print("Precision Positive: %.3f  Recall Positive:  %.3f") % (PrecisionP, RecallP)
print
print("Precision Neutral: %.3f  Recall  Neutral:  %.3f") % (PrecisionNtrl, RecallNtrl)
print
print("Precision Negative: %.3f  Recall Negative:  %.3f") % (PrecisionN, RecallN)
print
print
print("Accuracy:           %.4f  F-Score %.4f") % (Accuracy, Fscore)


statsarray = []
statsarray.append(PrecisionP)
statsarray.append(RecallP)

statsarray.append(PrecisionNtrl)
statsarray.append(RecallNtrl)

statsarray.append(PrecisionN)
statsarray.append(RecallN)

statsarray.append(Accuracy)
statsarray.append(Fscore)


statsarray2 = []
statsarray2.append(totPrecision)
statsarray2.append(totRecall)
statsarray2.append(Accuracy)
statsarray2.append(Fscore)

print

# Re-initialize variables and arrays for Boolean Multinonial Naive Bayes
allwords          = []   # Contains all words in aspect sentences with a rating
masterdict        = []   # Contains unique words in aspect sentences with a rating
masterdictcnt     = []   # Contains count of unique words corresponding to masterdict
masterdictpos     = []   # Contains all positive words in aspect sentences with a rating
masterdictposcnt  = []   # Contains count of all positive words in aspect sentences with a rating
masterdictneg     = []   # Contains all negative words in aspect sentences with a rating
masterdictnegcnt  = []   # Contains count of all negative words in aspect sentences with a rating
masterdictntrl    = []   # Contains all neutral words in aspect sentences with a rating
masterdictntrlcnt = []   # Contains count of all neutral words in aspect sentences with a rating
uniquewords       = []   # Contains unique words in the first dimension and the count in the second dim


# Update master dictionary of aspect sentences with values using Boolean Naive Bayes
print("Update Master Dictionary with Aspect Sentence words");
train      = int(doclen*.80)
teststart  = train+1
print
print("Train on fist 80 percent of Aspect Sentences: %i") % (train)
print
print("Test on last 20 percent of Aspect Sentence:  %i to %i") % (teststart,doclen)
testpos    = 0 
testneg    = 0
testntrl   = 0
testtotal  = 0
for i in range(train):
  # For boolean - if any word found in the sentence more than once
  # just update various arrays 1 time
  boolarray = []
  sentence = sentword[i]
  temp = sentence.split(",");
  l = len(temp)
  for j in range(l):
    l = len(temp)
    word = remove_characters(temp[j])
    word = remove_symbols(word)
    word = word.lower()
    word = word.replace(" ","")
    # Use below when running Boolean Multinomial classification
    findb = find_word1(word, boolarray)
    if (findb == "x"):
     boolarray.append(word)
     if (word != ''):
      testtotal = testtotal + 1
      findx = find_word2(word, masterdict)
      if (findx == "x"):
        masterdict.append(word)
        masterdictcnt.append(1)
      else:
        masterdictcnt[findx]+=1
      
      if (sentrating[i] >3):
        findx = find_word2(word, masterdictpos)
        if (findx == "x"):
          masterdictpos.append(word)
          masterdictposcnt.append(1)
        else:
          masterdictposcnt[findx]+=1
        
        testpos = testpos + 1

      if (sentrating[i] == 3):
        findx = find_word2(word, masterdictntrl)
        if (findx == "x"):
          masterdictntrl.append(word)
          masterdictntrlcnt.append(1)
        else:
          masterdictntrlcnt[findx]+=1
        
        testntrl = testntrl + 1

      if (sentrating[i] <= 2):
        findx = find_word2(word, masterdictneg)
        if (findx == "x"):
          masterdictneg.append(word)
          masterdictnegcnt.append(1)
        else:
          masterdictnegcnt[findx]+=1

        testneg = testneg + 1


print("Boolean Multinomial Naive Bayes Statistical Method of Classification")
print
print
testarray = []
uniquecnt = len(masterdict)
pwordcnt    = 0
ntrlwordcnt = 0
nwordcnt    = 0
print ("Statistics for Corpus:  %s") % (fullcorpus)
print
print("Total Aspect sentences: %i") % (doclen)
print("Total Aspect words: %d from Aspect Sentences") % (testtotal)
print("Total Unique Aspect words: %d from Aspect Sentences") % (uniquecnt)
print("Total Positive Aspect words: %i Total Neutral Aspect words %i Total Negative Aspect words %i") % (testpos, testntrl, testneg)
print
print("Train on fist 80 percent of Aspect Sentences: %i") % (train)
print
print("Test on last 20 percent of Aspect Sentence: %i to %i") %(teststart, doclen)
print
print
print("List of 15 unique word counts")
print("Word\tTotal\tPos\tNtrl\tNeg")
for m in range(15):
 pword = masterdict[m]
 findx = find_word2(pword, masterdictpos)
 if (findx == "x"):
   pwordcnt = 0
 else:
   pwordcnt = masterdictposcnt[findx] 

 findx = find_word2(pword, masterdictntrl)
 if (findx == "x"):
   ntrlwordcnt = 0
 else:
   ntrlwordcnt = masterdictntrlcnt[findx] 

 findx = find_word2(pword, masterdictneg)
 if (findx == "x"):
   nwordcnt = 0
 else:
   nwordcnt = masterdictnegcnt[findx] 

 print("%s\t%i\t%i\t%i\t%i") % (masterdict[m], masterdictcnt[m], pwordcnt, ntrlwordcnt, nwordcnt)
 

print
print

#classpospr = (testpos/doclen)
#classnegpr = (testneg/doclen)
# Revised 12/13/16 - To calculate Prior Class Probability based on the Total Class words/ All words

classpospr  = (testpos/testtotal)
classntrlpr = (testntrl/testtotal)
classnegpr  = (testneg/testtotal)

print
print("Prior Probability Class Positive: %.3f") % (classpospr)
print("Prior Probability Class Neutral: %.3f") %  (classntrlpr)
print("Prior Probability Class Negative: %.3f") % (classnegpr)
print
print("Boolean Multinimial Naive Bayes Method")
print
print("ClassNaiveBayes = argmaxP(C)*Product(i=1 to n)P(W(i)|C)")
print
print("P(C) is the Prior probabilty of a Class given the Training Set")
print("P(C) = All words in a Class over the Total number of words in the Training Set")
print
print("P(W(i)|C) = Count(W(i),C) or the count of a specific word in a Class over all words in that Class")
print
print("The difference in this method is that if a word occurs more than once in a document")
print("it is counted it is given a binary representation instead of a fredquency")
print("The idea is that the frequency of a word in a sentence does not give us")
print("more information in terms of Sentiment")


print("Testing Boolean Multinomial Bayes on last 10 percent of Data")
print
print
pwordcnt    = 0
ntrlwordcnt = 0
nwordcnt    = 0
nbpos       = 0
ntrlpos     = 0
nbneg       = 0
pper        = 0
ntrlper     = 0
nper        = 0
pperarray   = []
ntrlperarray= []
nperarray   = []
#matcharray  = []
tp         = 0 # True  Positive
tn         = 0 # True  Negative
fp         = 0 # False Positive  - Classifiy as Possitive but Actually Negative or Neutral
fn         = 0 # False Negativce - Classify as Negative but Actually Positive or Neutral
tntrl      = 0 # True  Neutral    - 
fntrl      = 0 # False Neutral   - Classify as Neutral but Actually Positive or Negative


for i in range(teststart,doclen):
  pperarray   = [] # Array containing positive word probabilities
  ntrlperarray= [] # Array containing neutral word probabilities
  nperarray   = [] # Array containing negative word probabilities

  sentence = sentword[i]
  temp = sentence.split(",")
  l = len(temp)
  findword = 0
  for x in range(l):
    word = remove_characters(temp[x])
    word = remove_symbols(word)
    word = word.lower()
    if (word != ''):

    # 12/14/16 - Add Laplace smoothing
    # Add 1 to the count and the length of the vocabulary to the denominator

      findx = find_word2(word, masterdictpos)
      if (findx == "x"):
        pwordcnt = 1
      else:
        pwordcnt = masterdictposcnt[findx]
        pwordcnt+=1 

      pper = pwordcnt/(testpos + testtotal)
      pperarray.append(pper)

      findx = find_word2(word, masterdictntrl)
      if (findx == "x"):
        ntrlwordcnt = 1
      else:
        ntrlwordcnt = masterdictntrlcnt[findx] 
        ntrlwordcnt+=1

      if (testntrl !=0):
         ntrlper = ntrlwordcnt/(testntrl + testtotal)
      else:
         ntrlper =0

      ntrlperarray.append(ntrlper)

    
      findx = find_word2(word, masterdictneg)
      if (findx == "x"):
        nwordcnt = 1
      else:
        nwordcnt = masterdictnegcnt[findx]
        nwordcnt+=1

      nper = nwordcnt/(testneg + testtotal)
      nperarray.append(nper)
  

  naivebayespos  = np.prod(np.array(pperarray))
  naivebayesntrl = np.prod(np.array(ntrlperarray))
  naivebayesneg  = np.prod(np.array(nperarray))

  nbpos = (classpospr  * naivebayespos)
  nbntrl= (classntrlpr * naivebayesntrl)
  nbneg = (classnegpr  * naivebayesneg) 

  print
  print("Sentiment Classification: Review Rating: %d : Aspect Sentence: %s") % (sentrating[i],sentword[i])
  print
  print("Boolean Multinomial Naive Bayes Method")
  if (nbpos > nbneg):
    print("Classified:Positive")
    if (sentrating[i]>3):
       print("Sentence Rating Positive: Correct Classification")
       print("Aspect sentence value %i") % (sentrating[i])
       tp=tp+1 

  
    if (sentrating[i]==3):
       print("Sentence Rating Neutral: Incorrect Classification")
       print("Aspect setient value %i") % (sentrating[i])
       fp+=1


    if (sentrating[i]<=2):
       print("Sentence Rating Negative: Incorrect Classification")
       print("Aspect sentence value %i") % (sentrating[i])
       fp+=1

  if (nbpos < nbneg):
    print("Classified: Negative")
    if (sentrating[i] <=2):
       print("Sentence Rating Negative: Correct Classification")
       print("Aspect sentence value %i") % (sentrating[i])
       tn+=1

    
    if (sentrating[i]==3):
       print("Sentence Rating Neutral: Incorrect Classification")
       print("Aspect sentence value %i") % (sentrating[i])
       fn+=1

    if (sentrating[i]>3):
       print("Sentence Rating Positive: Incorrect Classification")
       print("Aspect sentence value %i") % (sentrating[i])
       fn+=1

  if (nbpos == nbneg):
    print("Classified: Neutral")
    if (sentrating[i] == 3):
       print("Sentence Rating Neutral: Correct Classification")
       print("Aspect setient value %i") % (sentrating[i])
       tntrl+=1 
          
    if (sentrating[i] <=2):
       print("Sentence Rating Negative: Incorrect Classification")
       print("Aspect setient value %i") % (sentrating[i])
       fntrl+=1
      
    if (sentrating[i]>3):
       print("Sentence Rating Positive: Incorrect Classification")
       print("Aspect sentence value %i") % (sentrating[i])
       fntrl+=1

   
print
print("Boolean Multinomial Naive Bayes Statistical Method of Classification")
print("Evaluation")
print
print
print("True Positive:  %i     False Positive %i")  % (tp, fp)
print("False Negative: %i     True  Negative %i") %  (fn ,tn)
print("True Neutral:   %i     False Neutral  %i")  % (tntrl, fntrl)

evalstats.append(tp)
evalstats.append(fp)
evalstats.append(tn)
evalstats.append(fn)

if ((tp >0) or (fp > 0)):
    PrecisionP    = tp / (tp + fp)
else:
    PrecisionP  = 0

if ((tp > 0) or (fn > 0)):
   RecallP       = tp / (tp + fn)
else:
   RecallP = 0


if ((tntrl >0) or (fntrl > 0)):
    PrecisionNtrl   = tntrl / (tntrl + fntrl)
else:
    PrecisionNtrl  = 0

if ((tntrl > 0) or (fntrl > 0)):
   RecallNtrl       = tntrl / (tntrl + fntrl)
else:
   RecallNtrl = 0


if ((tn >0) or (fn > 0)):
    PrecisionN    = tn / (tn + fn)
else:
    PrecisionN = 0

if ((tn > 0) or (fp > 0)):
    RecallN       = tn / (tn + fp)
else:
    RecallN  = 0


Accuracy     = (tp + tn + tntrl)/(tp + tn + tntrl + fp + fn + fntrl)

totPrecision = PrecisionP + PrecisionN + PrecisionNtrl
totRecall    = RecallP    + RecallN    + RecallNtrl



if ((totPrecision > 0) or (totRecall > 0)):
   Fscore       = (2*totPrecision*totRecall)/(totPrecision + totRecall)
else:
   Fscore = 0

print
print("Accuracy in the context of Classification")
print("The total number of correctly predicted over all records")
print
print("Precision in the context of Classification:")
print("The ratio of correct predictions in a Class")
print("over all predictions for that Class: Correct and Incorrect")
print("It is a measure of exactness")
print
print("Recall in the Context of Classification:")
print("The number of correct predictions for a Class")
print("over the number of items actually belonging to that Class")
print("based on observations")
print("It is a measure of completness")
print
print("F-score balances Precision and Recall giving equal weight to both")
print
print("Observation for Book Reviews: Rating")
print("Ratings of 1-2 are negative")
print("Ratings equal to 3 are neutral")
print("Ratings of 4-5 are positive")
print
print("Observations for Free Text: Word Sentence Sentiment Score")
print


print("Precision Positive: %.3f  Recall Positive: %.3f") % (PrecisionP, RecallP)
print
print("Precision Neutral: %.3f  Recall  Neutral:  %.3f") % (PrecisionNtrl, RecallNtrl)
print
print("Precision Negative: %.3f  Recall Negative: %.3f") % (PrecisionN, RecallN)
print
print("Accuracy:           %.4f  F-Score %.4f") % (Accuracy, Fscore)

statsarray.append(PrecisionP)
statsarray.append(RecallP)

statsarray.append(PrecisionNtrl)
statsarray.append(RecallNtrl)

statsarray.append(PrecisionN)
statsarray.append(RecallN)

statsarray.append(Accuracy)
statsarray.append(Fscore)

statsarray2.append(totPrecision)
statsarray2.append(totRecall)
statsarray2.append(Accuracy)
statsarray2.append(Fscore)

print
print
print("Comparison of Naive Bayes and Boolean Multinomial Naive Bayes")
print("Positive Precision: %.3f       %.3f") % (statsarray[0], statsarray[8])
print("Neutral  Precision: %.3f       %.3f") % (statsarray[2], statsarray[10])
print("Negative Precision: %.3f       %.3f") % (statsarray[4], statsarray[12])
print("Positive Recall:    %.3f       %.3f") % (statsarray[1], statsarray[9])
print("Neutral  Recall:    %.3f       %.3f") % (statsarray[3], statsarray[11])
print("Negative Recall:    %.3f       %.3f") % (statsarray[5], statsarray[13])
print("Accuracy:           %.3f       %.3f") % (statsarray[6], statsarray[14])
print("F-score  Positive:  %.3f       %.3f") % (statsarray[7], statsarray[15])




print
print("Find Key Word Aspect in each Sentence")
# Get Sysnet features for each trimmed sentence
print
print
featuresets=[]
for i in range(keycnt):
  keyword = keywords[i]
  for j in range(doclen):
    synsent  = ""
    sentence = sentword[j]
    rating   = sentrating[j]
    title    = senttitle[j]
    temp = sentence.split(",");
    l = len(temp)
    findword = 0
    for x in range(l):
        word = remove_characters(temp[x])
        word = remove_symbols(word)
        word = word.lower()
        if (word != ''):
         if (word == keyword):
              findword=1
              break

    if (findword == 1):
      phrase   = lesk(sentence,keyword)
      findx = find_phrase(phrase, syns)
      if (findx != "x"):
        synsent = synsdef[findx]
        synsent = synsent.encode('ascii')
        key = ("%s\t %s") % (keyword,sentence)
        aspects.append(keyword)
        aspectsent.append(sentence)
        aspectrating.append(rating)
        aspecttitle.append(title)
        s = synsname[findx]
        results1 = swn.senti_synset(s)
        pscore = results1.pos_score()
        nscore = results1.neg_score()
        temp = ("Synset: %s\tPos: %.2f\tNeg: %.2f")  % (syns[findx],pscore,nscore)
        aspectsynval.append(temp)
        aspectsynpos.append(pscore)
        aspectsynneg.append(nscore)
    


  

totalsentimentvalue = 0
sentimentval        = 0
dl                  = len(aspectsent)

for i in range(dl):
  temp = evaluate_sentences(aspectsent[i])
  aspectval.append(temp)

 
  
   
print
print("Corpus Sentence with keyword Aspect Count: %s") % (dl)
print
doclen = len(sentrating)


print
print("Bag of Words Classification Model")
# Initialize variables
aspectcnt    = 0 # Total Aspect sentences
totsentval   = 0 # Total Aspect sentence sentiment value
totratingval = 0 # Total Aspect sentence rating value

aspectneg    = 0 # Total negative aspect sentences
aspectpos    = 0 # Total positive aspect sentences

allpossent   = 0 # Grand Total positive sentences
allnegsent   = 0 # Grand Total negative sentences
gtotsent     = 0 # Grand Total pos and neg sentences
gtotratingval= 0 # Grand Total aspect sentence ratings

tp           = 0 # True Positive
tn           = 0 # True Negative
fp           = 0 # False Positive  - Classifiy as Possitive but Actually Negative
fn           = 0 # False Negativce - Classify as Negative but Actually Positive
tntrl        = 0 # True Neutral 
fntrl       = 0 # False Neutral   -  Classify as Neutrail but actually Negative or Positive

print
oldaspect = ""
for i in range(dl):
  if (i == 0):
   oldaspect = aspects[i]
   aspectcnt = 0
   print("Aspect : %s") % (oldaspect)
   print

  if (oldaspect != aspects[i]):
    print
    print("Total Sentences for Aspect: %s : %i Overall Sentiment Value: %i") % (oldaspect, aspectcnt, totsentval)
    posper = (aspectpos / aspectcnt)
    negper = (aspectneg / aspectcnt)
    avgrating = (totratingval / aspectcnt)
    print("Number Classified Positive: %i       Number Classfied Negative: %i")    % (aspectpos, aspectneg)
    print("Percent Positive:           %.2f     Percent Negative:          %.2f")  % (posper, negper)
    print("Average Aspect Review Rating:                                  %.4f")  % (avgrating)
    allpossent = allpossent + aspectpos
    allnegsent = allnegsent + aspectneg
    oldaspect = aspects[i]
    aspectcnt=0
    totsentval = 0
    aspectneg = 0
    aspectpos = 0
    gtotratingval= gtotratingval + totratingval
    totratingval = 0 
    print
    print("Aspect : %s") % (oldaspect)
    print

  if (aspectval[i] != 0):
    aspectcnt=aspectcnt + 1
    print
    #print("Book Title:  %s ") % (aspecttitle[i])
    print("Review Rating: %.2f Sentinment Value: %i Sentence: %s") % (aspectrating[i], aspectval[i], aspectsent[i])
    totsentval  = totsentval + aspectval[i]
    totratingval= totratingval + aspectrating[i]
    print
    if (aspectval[i] > 0): 
       aspectpos = aspectpos + 1
       #print("Classified:Positive")
       #print
       if (aspectrating[i]>=3):
         #print("Sentence Rating Positive: Correct Classification")
         #print("Aspect sentence value %i Aspect Rating %.2f") % (aspectval[i], aspectrating[i])
         tp=tp+1 

       if (aspectrating[i]<=2):
         #print("Sentence Rating Negative: Incorrect Classification")
         #print("Aspect sentence value %i Aspect Rating %.2f") % (aspectval[i], aspectrating[i])
         fp+=1

   
    if (aspectval[i] < 0): 
       aspectneg = aspectneg + 1
       #print("Classified:Negative")
       #print
       if (aspectrating[i]<=2):
         #print("Sentence Rating Negative: Correct Classification")
         #print("Aspect sentence value %i Aspect Rating %.2f") % (aspectval[i], aspectrating[i])
         tn=tn+1 
 
       if (aspectrating[i]>=3):
         #print("Sentence Rating Positive: InCorrect Classification")
         #print("Aspect sentence value %i Aspect Rating %.2f") % (aspectval[i], aspectrating[i])
         fn=fn+1 
   

    
    if ((aspectsynpos[i] != 0) or (aspectsynneg[i] != 0)):
      print(aspectsynval[i])


     
if (oldaspect == aspects[i]):
    print
    print("Total Sentences for Aspect: %s : %i  Overall Sentiment Value: %i") % (oldaspect, aspectcnt, totsentval)
    posper = (aspectpos / aspectcnt)
    negper = (aspectneg / aspectcnt)
    gtotratingval = gtotratingval + totratingval
    avgrating = (totratingval / aspectcnt)
    allpossent = allpossent + aspectpos
    allnegsent = allnegsent + aspectneg
    print("Number Classified Positive: %i       Number Classfied Negative: %i")    % (aspectpos, aspectneg)
    print("Percent Positive:           %.2f     Percent Negative:          %.2f")  % (posper, negper)
    print("Average Aspect Review Rating:                                   %.2f")  % (avgrating)
    oldaspect = aspects[i]
    print
 

  
print
print
gtotsent = allpossent + allnegsent
print("Grand Total Sentences: %i") % (gtotsent)
posper = (allpossent / gtotsent)
negper = (allnegsent/ gtotsent)
avgrating = (gtotratingval / gtotsent)
print("Number Classified Positive: %i     Number Classified Negative: %i")  % (allpossent, allnegsent)
print("Percent Positive:           %.2f     Percent Negative:         %.2f" )  % (posper, negper)
print("Average Aspect Rating:                                         %.2f")   % (avgrating)

print
print

print
print("Bag of Words Classification Model")
print("Evaluation")
print
print
print("True Positive:  %i     False Positive %i")  % (tp, fp)
print("False Negative: %i     True  Negative %i") % (fn ,tn)

evalstats.append(tp)
evalstats.append(fp)
evalstats.append(tn)
evalstats.append(fn)
print
if ((tp >0) or (fp > 0)):
    PrecisionP    = tp / (tp + fp)
else:
    PrecisionP  = 0

if ((tp > 0) or (fn > 0)):
   RecallP       = tp / (tp + fn)
else:
   RecallP = 0


if ((tn >0) or (fn > 0)):
    PrecisionN    = tn / (tn + fn)
else:
    PrecisionN = 0

if ((tn > 0) or (fp > 0)):
    RecallN       = tn / (tn + fp)
else:
    RecallN  = 0

Accuracy     = (tp + tn)/(tp + tn + fp + fn)

totPrecision = PrecisionP + PrecisionN
totRecall    = RecallP    + RecallN

if ((totPrecision > 0) or (totRecall > 0)):
  Fscore       = (2*totPrecision*totRecall)/(totPrecision + totRecall)
else:
  Fscore = 0

print
print("Accuracy in the context of Classification")
print("The total number of correctly predicted over all records")
print
print
print("Precision in the context of Classification:")
print("The ratio of correct predictions in a Class")
print("over all predictions for that Class: Correct and Incorrect")
print("It is a measure of exactness")
print
print("Recall in the Context of Classification:")
print("The number of correct predictions for a Class")
print("over the number of items actually belonging to that Class")
print("based on observations")
print("It is a measure of completness")
print
print("F-score balances Precision and Recall giving equal weight to both")
print
print
print("Observation for Book Reviews: Rating")
print("Ratings of 1-2 are negative")
print("Ratings of 3-5 are positive")
print
print("Observations for Free Text: Word Sentence Sentiment Score")
print

print("Precision Positive: %.3f  Recall Positive:  %.3f") % (PrecisionP, RecallP)
print
print("Precision Negative: %.3f  Recall Negative:  %.3f") % (PrecisionN, RecallN)
print
print("Accuracy:           %.4f  F-Score %.4f") % (Accuracy, Fscore)


statsarray.append(PrecisionP)
statsarray.append(RecallP)
statsarray.append(PrecisionN)
statsarray.append(RecallN)
statsarray.append(Accuracy)
statsarray.append(Fscore)

statsarray2.append(totPrecision)
statsarray2.append(totRecall)
statsarray2.append(Accuracy)
statsarray2.append(Fscore)

print
print
print
print("Comparison of Naive Bayes      \tBoolean MN\tBag of Words")
print("                                Naive Bayes              ")
print("Positive Precision: %.3f       \t%.3f        \t%.3f") % (statsarray[0], statsarray[8], statsarray[16])
print("Neutral  Precision: %.3f       \t%.3f        \t")     % (statsarray[2], statsarray[10])
print("Negative Precision: %.3f       \t%.3f        \t%.3f") % (statsarray[4], statsarray[12],statsarray[18])
print("Positive Recall:    %.3f       \t%.3f        \t%.3f") % (statsarray[1], statsarray[9], statsarray[17])
print("Neutral  Recall:    %.3f       \t%.3f        \t")     % (statsarray[3], statsarray[11])
print("Negative Recall:    %.3f       \t%.3f        \t%.3f") % (statsarray[5], statsarray[13], statsarray[19])
print("Accuracy:           %.3f       \t%.3f        \t%.3f") % (statsarray[6], statsarray[14], statsarray[20])
print("F-score:            %.3f       \t%.3f        \t%.3f") % (statsarray[7], statsarray[15], statsarray[21])



# Need to sort both array in the first axis
print
print("Summary Statistics for 3 models")
print("Naive Bayes and Boolean Naive Bayes are First with an Accuracy of  %.3f") % (statsarray[6])
print("Bag of Words Model is third with an Accuracy of                    %.3f") % (statsarray[20])
print
print
print("Bag of words model has a better F-score with                       %.3f") % (statsarray[21])
print("Naive Bays and Multinomial Naive Bayes are tied with an F-Score of %.3f") % (statsarray[15])
print
print
print
print
print("Detail Statistics for 3 Models True and False Positive and Negative")
print("Naive Bayes")
print
print("True Positive:  %i     False Positive %i")  % (evalstats[0], evalstats[1])
print("False Negative: %i      True  Negative %i") %  (evalstats[2], evalstats[3])
print
print("Boolean Multinomial Naive Bayes")
print
print("True Positive:  %i     False Positive %i")  % (evalstats[4], evalstats[5])
print("False Negative: %i      True  Negative %i") %  (evalstats[6], evalstats[7])
print
print("Bag of Words")
print
print("True Positive:  %i     False Positive %i")  % (evalstats[8], evalstats[9])
print("False Negative: %i      True  Negative %i") %  (evalstats[10], evalstats[11])
print
