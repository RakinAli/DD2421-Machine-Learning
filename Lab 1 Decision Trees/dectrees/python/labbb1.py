import monkdata as m
import dtree as d
import drawtree_qt5 as draw
import matplotlib.pyplot as plt

import random


data_sets = [m.monk1, m.monk1test, m.monk2, m.monk2test, m.monk3, m.monk3test]

# Assignment 1
print("Assignment 1")
print(d.entropy(m.monk1))
print(d.entropy(m.monk2))
print(d.entropy(m.monk3))


# Assignment 3
print("Assignment 3")
print("Monk1")
print(round(d.averageGain(m.monk1, m.attributes[0]), 5), round(d.averageGain(m.monk1, m.attributes[1]), 5), round(d.averageGain(m.monk1, m.attributes[2]), 5), round(
    d.averageGain(m.monk1, m.attributes[3]), 5), round(d.averageGain(m.monk1, m.attributes[4]), 5), round(d.averageGain(m.monk1, m.attributes[5]), 5))
print("Monk2")
print(round(d.averageGain(m.monk2, m.attributes[0]), 5), round(d.averageGain(m.monk2, m.attributes[1]), 5), round(d.averageGain(m.monk2, m.attributes[2]), 5), round(
    d.averageGain(m.monk2, m.attributes[3]), 5), round(d.averageGain(m.monk2, m.attributes[4]), 5), round(d.averageGain(m.monk2, m.attributes[5]), 5))
print("Monk3")
print(round(d.averageGain(m.monk3, m.attributes[0]), 5), round(d.averageGain(m.monk3, m.attributes[1]), 5), round(d.averageGain(m.monk3, m.attributes[2]), 5), round(
    d.averageGain(m.monk3, m.attributes[3]), 5), round(d.averageGain(m.monk3, m.attributes[4]), 5), round(d.averageGain(m.monk3, m.attributes[5]), 5))

# Assignment 4
print("Assignment 4")
print("Monk1")
print(d.bestAttribute(m.monk1, m.attributes))
print("Monk2")
print(d.bestAttribute(m.monk2, m.attributes))
print("Monk3")
print(d.bestAttribute(m.monk3, m.attributes))

# Assignment 5
print("Assignment 5")
print("Monk1")

# print(d.check(t, m.monk1))

print("Training error Monk1: ")
print(1 - d.check(d.buildTree(m.monk1, m.attributes), m.monk1))

print("Test error Monk1: ")
print(1 - d.check(d.buildTree(m.monk1, m.attributes), m.monk1test))

print("Training error Monk2: ")
print(1 - d.check(d.buildTree(m.monk2, m.attributes), m.monk2))

print("Test error Monk1: ")
print(1 - d.check(d.buildTree(m.monk2, m.attributes), m.monk2test))

print("Training error Monk3: ")
print(1 - d.check(d.buildTree(m.monk3, m.attributes), m.monk3))

print("Test error Monk3: ")
print(1 - d.check(d.buildTree(m.monk3, m.attributes), m.monk3test))

# print(d.allPruned(t))


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


monk1train, monk1val = partition(m.monk1, 0.8)
print("Length of training data when frac is 0.3:", len(monk1train))
print("Length of validation data when frac is 0.3:", len(monk1val))


def pruning(monktest, monktrain, monkval):
    best_tree = d.buildTree(monktrain, m.attributes)
    best_error = 1 - d.check(best_tree, monkval)
    for tree in d.allPruned(best_tree):
        error = 1 - d.check(tree, monkval)
        if error < best_error:
            best_error = error
            best_tree = tree
    return 1 - d.check(best_tree, monktest), best_tree


error_monk1, tree_monk1 = pruning(m.monk1test, monk1train, monk1val)

# run pruning 100 times and calculate the average error
error_monk1 = 0
for i in range(1000):
    monk1train, monk1val = partition(m.monk1, 0.3)
    error_monk1 += pruning(m.monk1test, monk1train, monk1val)[0]
error_monk1 /= 1000


unpruned_tree = d.buildTree(monk1train, m.attributes)
error_monk1_unpruned = 0
for i in range(1000):
    err = 1 - d.check(unpruned_tree, monk1val)
    error_monk1_unpruned += err
error_monk1_unpruned /= 1000

fraction_arr = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
monk1_pruned_err = [0, 0, 0, 0, 0, 0]
monk1_unpruned_err = [0, 0, 0, 0, 0, 0]
monk3_pruned_err = [0, 0, 0, 0, 0, 0]
monk3_unpruned_err = [0, 0, 0, 0, 0, 0]
tot_monk1_pruned_err = 0
tot_monk1_unpruned_err = 0
tot_monk3_pruned_err = 0
tot_monk3_unpruned_err = 0
for i in range(6):
    for j in range(1000):
        monk1train, monk1val = partition(m.monk1, fraction_arr[i])
        tot_monk1_pruned_err += pruning(m.monk1test, monk1train, monk1val)[0]
        tot_monk1_unpruned_err += 1 - \
            d.check(d.buildTree(monk1train, m.attributes), monk1val)

        monk3train, monk3val = partition(m.monk3, fraction_arr[i])
        tot_monk3_pruned_err += pruning(m.monk3test, monk3train, monk3val)[0]
        tot_monk3_unpruned_err += 1 - \
            d.check(d.buildTree(monk3train, m.attributes), monk3val)
    monk1_pruned_err[i] = tot_monk1_pruned_err / 1000
    tot_monk1_pruned_err = 0
    monk1_unpruned_err[i] = tot_monk1_unpruned_err / 1000
    tot_monk1_unpruned_err = 0
    monk3_pruned_err[i] = tot_monk3_pruned_err / 1000
    tot_monk3_pruned_err = 0
    monk3_unpruned_err[i] = tot_monk3_unpruned_err / 1000
    tot_monk3_unpruned_err = 0

print("Monk1 pruned error array: ", monk1_pruned_err)
print("Monk1 unpruned error array: ", monk1_unpruned_err)
print("Monk3 pruned error array: ", monk3_pruned_err)
print("Monk3 unpruned error array: ", monk3_unpruned_err)


plt.figure(figsize=(10, 5), dpi=80)
csfont = {'fontname': 'Comic Sans MS'}
plt.plot(fraction_arr, monk1_pruned_err, color="blue",
         linewidth=2.5, linestyle="-", label="monk1 pruned")
plt.plot(fraction_arr, monk1_unpruned_err, color="red",
         linewidth=2.5, linestyle="-", label="monk1 unpruned")
# Show the data points a part from the line plot
plt.scatter(fraction_arr, monk1_pruned_err, color="blue",
            linewidth=2.5, linestyle="dotted")
plt.scatter(fraction_arr, monk1_unpruned_err, color="red",
            linewidth=2.5, linestyle="dotted")
plt.ylabel("Error rate", **csfont)
plt.xlabel("Fraction", **csfont)
plt.title("Monk 1", **csfont, fontsize=10, color='blue')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10, 5), dpi=80)
csfont = {'fontname': 'Comic Sans MS'}
plt.plot(fraction_arr, monk3_pruned_err, color="blue",
         linewidth=2.5, linestyle="-", label="monk3 pruned")
plt.plot(fraction_arr, monk3_unpruned_err, color="red",
         linewidth=2.5, linestyle="-", label="monk3 unpruned")
# Show the data points a part from the line plot
plt.scatter(fraction_arr, monk3_pruned_err, color="blue",
            linewidth=2.5, linestyle="dotted")
plt.scatter(fraction_arr, monk3_unpruned_err, color="red",
            linewidth=2.5, linestyle="dotted")
plt.ylabel("Error rate", **csfont)
plt.xlabel("Fraction", **csfont)
plt.title("Monk 3", **csfont, fontsize=10, color='blue')
plt.legend(loc='upper right')
plt.show()

t = d.buildTree(m.monk2, m.attributes)
