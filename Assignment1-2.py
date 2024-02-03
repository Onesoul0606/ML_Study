# these are the libraries we are going to use
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from RM1985 import *
matplotlib.style.use('seaborn-v0_8-notebook')
import warnings
warnings.filterwarnings('ignore')
np.random.seed(2222)


#Encoding the input
word_to_wickelphone('kAm')
activate_word('kAm')

# Exercise 1a)
# Let's translate the ten high frequency words used in the paper:
# come/came, look/looked, feel/felt, have/had, make/made, get/got, give/gave, take/took, go/went, like/liked
high_frequency_verbs = ['come', 'look', 'feel', 'have', 'make', 'get', 'give', 'take', 'go', 'like']
base_high_frequency_verbs = ['k*m', 'luk', 'fEl', 'hav', 'mAk', 'get', 'giv', 'tAk', 'gO', 'lIk' ]
past_high_frequency_verbs = ['kAm', 'lukt', 'felt', 'had','mAd', 'got', 'gAv', 'tuk','went', 'likt' ]

#Exercise 1b)
#Store the shape of the wickelfeature represetation for came and the number of wickelfeatures activated (i.e., set to 1).
came_activation = np.array(activate_word('kAm'))
came_shape = np.shape(came_activation)
print(came_shape) #(460,)
came_number_active_wickelfeatures = np.count_nonzero(came_activation)
print(came_number_active_wickelfeatures) #16

#Exercise 2a)
#Translate this function into code.
#1/1+𝑒^(−(𝑛𝑒𝑡−𝜃)/𝑇)
def rm_activation_function(net, theta, T=1.0):
    probability = 1/(1 + np.exp(-(net - theta)/T))
    return probability

net_activation = np.arange(-5, 6)
p_T1 = rm_activation_function(net_activation, theta=0.0, T=1.0)
p_T05 = rm_activation_function(net_activation, theta=0.0, T=0.5)
p_T2 = rm_activation_function(net_activation, theta=0.0, T=2.0)

#Exercise 2b)
#Let's set theta = 0.0 and plot the probability of firing as a function of the the weighted activation 
# net at T=1.0, T=0.5 and T=2.0 in one figure. We should consider weighted activation values between -5 and 5.
plt.figure()
plt.title("RM's Perceptron")
plt.plot(net_activation, p_T1, label='T=1.0', color='red')
plt.plot(net_activation, p_T05, label='T=0.5', color='blue')
plt.plot(net_activation, p_T2, label='T=2.0', color='green')
plt.xlabel("Weighted Activation")
plt.ylabel("Probability of Firing")
plt.xlim(-5,5)
plt.ylim(0,1)
plt.legend()
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()

#U Shaped Curves
verbs = []
with open('c:\gitLocal\ML\\verbs.csv') as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            print(line.strip('\n').split(','))
        else:
            verbs.append(line.strip('\n').split(','))
            print(verbs[-1])

# add in the high frequency verbs you translated for us earlier
for i, word in enumerate(high_frequency_verbs):
    if word in ['look', 'like']:
        verbs.append([word, 'Regular', base_high_frequency_verbs[i], past_high_frequency_verbs[i], 'H'])
    else:
        verbs.append([word, 'Irregular', base_high_frequency_verbs[i], past_high_frequency_verbs[i], 'H'])
    #print(verbs[-1])
    
base_wickel_HF = np.array([activate_word(w) for w in base_high_frequency_verbs]).T #.T: Transpose
past_wickel_HF = np.array([activate_word(w) for w in past_high_frequency_verbs]).T

percept = Perceptron(active=rm_activation_function)

percept.learn(base_wickel_HF, past_wickel_HF)
#array([0.97173913, 0.94782609, 0.93695652, 0.97173913, 0.95652174,
#       0.96956522, 0.95      , 0.96304348, 0.84347826, 0.93043478])

percept.score(base_wickel_HF, past_wickel_HF)
#array([0.9673913 , 0.95434783, 0.94347826, 0.9673913 , 0.95869565,
#       0.97391304, 0.9673913 , 0.96304348, 0.87173913, 0.92826087])


#Exercise 3a)
#Now let's divide the corpus verbs into two lists: 
# one for regular verbs and one for irregular verbs.
regular_verbs = []
irregular_verbs = []
for i in verbs:
    if i[1] == "Regular":
        regular_verbs.append(i)
    elif i[1] == "Irregular":
        irregular_verbs.append(i)
print(len(regular_verbs))
print(len(irregular_verbs))
#Exercise 3b)
#First convert the phonemes for base and past tense into wickelfeatures. 
# Then calculate the mean score of the model on irregular and regular verbs.
base_wickel_irregular = np.array([activate_word(w[2]) for w in irregular_verbs]).T
past_wickel_irregular = np.array([activate_word(w[3]) for w in irregular_verbs]).T
base_wickel_regular = np.array([activate_word(w[2]) for w in regular_verbs]).T
past_wickel_regular = np.array([activate_word(w[3]) for w in regular_verbs]).T

irregular_score = np.mean(percept.score(base_wickel_irregular, past_wickel_irregular))
regular_score = np.mean(percept.score(base_wickel_regular, past_wickel_regular))
print(irregular_score)
print(regular_score)

# Let's initialize a new perceptron with our custom activation function
percept = Perceptron(active=rm_activation_function)

# Now let's loop through each data point to train and score
scores_regular = []
scores_irregular = []
for i in range(len(high_frequency_verbs)):
    percept.learn(base_wickel_HF[:,i, np.newaxis], past_wickel_HF[:,i,np.newaxis])
    scores_regular.append(np.mean(percept.score(base_wickel_regular, past_wickel_regular)))
    scores_irregular.append(np.mean(percept.score(base_wickel_irregular, past_wickel_irregular)))
    print(scores_regular[-1], scores_irregular[-1])

#Exercise 4a)
#First, we need to extract the medium frequency verbs from the corpus verbs
base_med_frequency_verbs = []
past_med_frequency_verbs = []
for i in verbs:
    if i[4] == 'M':
        base_med_frequency_verbs.append(i[2])
        past_med_frequency_verbs.append(i[3])
print(base_med_frequency_verbs)
print(past_med_frequency_verbs)  

#Exercise 4b)
#Second, we need to convert those those verbs into wickelfeatures.
base_wickel_MF = np.array([activate_word(w) for w in base_med_frequency_verbs]).T
past_wickel_MF = np.array([activate_word(w) for w in base_med_frequency_verbs]).T

#Exercise 4c)
#Calculate and store the scores for regular and irregular verbs in the variables scores_irregular_md and scores_regular_md.
scores_regular_md = []
scores_irregular_md = []
for i in range(len(base_med_frequency_verbs)):
    base_wickel_med_frequency_regular = []
    past_wickel_med_frequency_regular = []
    base_wickel_med_frequency_irregular = []
    past_wickel_med_frequency_irregular = [] 
    
    if base_wickel_MF[i] in base_wickel_regular:
        base_wickel_med_frequency_regular.append(base_wickel_MF[i])
    elif base_wickel_MF[i] in base_wickel_irregular:
        base_wickel_med_frequency_irregular.append(base_wickel_MF[i])
        
    percept.learn(base_wickel_HF[:,i, np.newaxis], past_wickel_HF[:,i,np.newaxis])
    scores_regular_md.append(np.mean(percept.score(base_wickel_MF, base_wickel_med_frequency_regular)))
    scores_irregular.append(np.mean(percept.score(base_wickel_irregular, past_wickel_irregular)))
    print(scores_regular_md[-1], scores_irregular_md[-1])
















