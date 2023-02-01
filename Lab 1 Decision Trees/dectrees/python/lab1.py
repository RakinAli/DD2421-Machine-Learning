import monkdata as m
import dtree as d

# Compute the entropy of the training data for each attribute - Assignment 1
print("Entropy of the training data for each attribute:")
print("A1: ", d.entropy(m.monk1))
print("A2: ", d.entropy(m.monk2))
print("A3: ", d.entropy(m.monk3))

# Assignment 3 - Compute the information gain of the training data for each attribute
print("Information gain of the training data for each attribute:")
for i in range(0, 6):
    print("A1: ", d.averageGain(m.monk1, m.attributes[i]))

for i in range(0, 6):
    print("A2: ", d.averageGain(m.monk2, m.attributes[i]))

for i in range(0, 6):
    print("A3: ", d.averageGain(m.monk3, m.attributes[i]))

# Assignment 4 - Förklara Entropi och information gain

# Assignment 5 - Build a decision tree for each training set and compute the training and test error
t1 = d.buildTree(m.monk1, m.attributes)
t2 = d.buildTree(m.monk2, m.attributes)
t3 = d.buildTree(m.monk3, m.attributes)

print("Training error for each tree:")
print("T1: ", 1 - d.check(t1, m.monk1))
print("T1: ", 1 - d.check(t1, m.monk1test))

print("T2: ", 1 - d.check(t2, m.monk2))
print("T2: ", 1 - d.check(t2, m.monk2test))

print("T3: ", 1 - d.check(t3, m.monk3))
print("T3: ", 1 - d.check(t3, m.monk3test))


# Assignment 6 - Prune the trees using the validation set
p1 = d.allPruned(t1)
p2 = d.allPruned(t2)
p3 = d.allPruned(t3)

print("Pruned trees:")
print("P1: ", p1)
print("P2: ", p2)
print("P3: ", p3)

print("Pruned training error for each tree:")
print("P1: ", 1 - d.check(p1, m.monk1))
print("P1: ", 1 - d.check(p1, m.monk1test))

print("P2: ", 1 - d.check(p2, m.monk2))
print("P2: ", 1 - d.check(p2, m.monk2test))

print("P3: ", 1 - d.check(p3, m.monk3))
print("P3: ", 1 - d.check(p3, m.monk3test))

# Assignment 7 - Which tree is the best on the test set?
