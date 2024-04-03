import numpy as np
from matplotlib import pyplot as plt

#address the input file in line 17

vocab = []  #vocabulary (unique words)
words = []  #all the words in text file
n=0   #number of words
v=0   #number of unique words
N=[]  #number of words read so far (updated after reading a new word not seen before)
V=[]  #number of unique words read so far (updated after reading a new word not seen before)
vocabcount=[]  #counts number of occurrences of words
"""
for heaps we go through the text and whenever we see a new word we add it to the dictionary and plot it--> x axis = number of words so far and y axis = number of unique words so far (vocabulary)
for zipfs we count all the words with their frequencies and calculate the probability
"""
# opening the text file
with open('sample3.txt', 'r') as file:
    # reading each line
    for line in file:

        # reading each word
        for word in line.split():
            n=n+1
            if word in vocab:         #if its not  a new word
                for m in range(len(vocab)):     #find the word and add to its frequency
                    if vocab[m]==word:
                        vocabcount[m]=vocabcount[m]+1
                        m = len(vocab)

            # displaying the words
            elif word not in vocab:  # if it is a new word add it to the vocabulary
                vocab.append(word)
                vocabcount.append(1)
                v=v+1
                N.append(n)   #when a new word is found, put the number of words at that point in N
                V.append(v)   # and put the number of unique points at that point in V
            words.append(word)

print('vocabulary size: ',len(vocab))
print('number of words: ',len(words))


#plot for heaps

fig, (ax2) = plt.subplots(1, 1, figsize=[7, 11])
Nlog , Vlog = np.log10(N) , np.log10(V)
m, b = np.polyfit(Nlog, Vlog, 1)
#plt.plot(Nlog, Vlog, 'o')

plt.plot(Nlog, m*Nlog + b)
ax2.plot(Nlog , Vlog)
#ax2.loglog(N,V)
ax2.set_title('loglog plot', fontsize=15)
ax2.set_xlabel('N, number of words', fontsize=13)
ax2.set_ylabel('V, vocabulary size', fontsize=13)
plt.tight_layout()
plt.show()
print("heaps fit line is: ",m,"*x + ",b)



#print("len vocabcount:",len(vocabcount))
#print("len vocab:",len(vocab))


zipped_lists = zip(vocabcount, vocab)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
vocabcount, vocab = [ list(tuple) for tuple in  tuples]

i = len(vocabcount)


vocab.reverse()
vocabcount.reverse()   #to be in increasing order
rank = []
for i in range(len(vocabcount)):
    rank.append(i+1)
vocabprob=[]
num = len(words)
for i in range(len(vocab)):
    x = vocabcount[i]/num
    vocabprob.append(x)



#plot for zipfs
fig, (ax2) = plt.subplots(1, 1, figsize=[7, 11])

ranklog , vocabcountlog = np.log10(rank) , np.log10(vocabcount)

m, b = np.polyfit(ranklog, vocabcountlog, 1)
#plt.plot(ranklog, vocabcountlog, 'o')

plt.plot(ranklog, m*ranklog + b)

ax2.plot(ranklog , vocabcountlog)
ax2.set_title('loglog plot', fontsize=15)
ax2.set_xlabel('rank', fontsize=13)
ax2.set_ylabel('frequency', fontsize=13)
plt.tight_layout()
plt.show()
print("zipfs fit line is: ",m,"*x + ",b)


















