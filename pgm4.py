import pandas as pd
from collections import Counter
import math

tennis = pd.read_csv('pgm4.csv')
print("\nGiven PlayTennis Data Set:\n\n", tennis)

def entropy(alist):
    c = Counter(x for x in alist)
    instances = len(alist)
    prob = [x/instances for x in c.values()]
    return sum([-p*math.log(p, 2) for p in prob])

def information_gain(d, split, target):
    splitting = d.groupby(split)
    n = len(d.index)
    agent = splitting.agg({target: [entropy, lambda x: len(x)/n]})
    agent.columns = ['Entropy', 'Observations']
    newentropy = sum(agent['Entropy'] * agent['Observations'])
    oldentropy = entropy(d[target])
    return oldentropy - newentropy

def id3(sub, target, a):
    count = Counter(x for x in sub[target])
    if len(count) == 1:
        return next(iter(count))
    else:
        gain = [information_gain(sub, attr, target) for attr in a]
        print("\nGain =", gain)
        maximum = gain.index(max(gain))
        best = a[maximum]
        print("\nBest Attribute:", best)
        tree = {best: {}}
        remaining = [i for i in a if i != best]
        for val, subset in sub.groupby(best):
            subtree = id3(subset, target, remaining)
            tree[best][val] = subtree
        return tree

names = list(tennis.columns)
print("\nList of Attributes:", names)
names.remove('PlayTennis')
print("\nPredicting Attributes:", names)

# Convert the 'observations' column to a dictionary
tree = id3(tennis, 'PlayTennis', names)
print("\n\nThe Resultant Decision Tree is:\n")
print(tree)