# CS124 programming assignment 3, the number partition problem

# imports
import random
import sys
import math
import time
import statistics


# globals
random.seed(2) # original, to be altered by code
fileSize = 100 # eventually to be 100
max_num = 1000000000000 # eventually to be made 10^12
max_iter = 250 # eventually to be 25000


# generate random input file with 100 numbers, one per line
def generateInputFile(fName = "input.txt", maxx = max_num):
    f = open(fName, 'w')
    for i in range(fileSize):
        f.write(str(random.randint(1, maxx)) + "\n")


# Karmarkar-Karp
def kK(arr):

    # so that we don't alter the original
    arr2 = [x for x in arr]

    # iterate from the top and go down until i = 0
    for i in range(len(arr2) - 1, 0, -1):

        # sort in ascending order
        arr2.sort()

        # residual after taking largest 2 numbers out of array
        residue = arr2[i] - arr2[i-1]
        arr2.remove(arr2[i])
        arr2.remove(arr2[i-1])

        # append residue back into array
        arr2.append(residue)

    return arr2[0]


# Repeated Random Standard Representation
def rRsr(arr, iterations = max_iter):

    # set current values
    current_solution = srs()
    best_solution = current_solution
    current_residue = standard_residue(arr, current_solution)

    # go through iterations - 1 (because did the first iteration above)
    for i in range(iterations - 1):

        # if new solution is better than old one, update
        current_solution = srs()
        new_residue = standard_residue(arr, current_solution)
        if new_residue < current_residue:
            best_solution = current_solution
            current_residue = new_residue

    # return solution array as well as final residue
    return best_solution, current_residue


# Repeated Random Prepartitioned Representation
def rRpr(arr, iterations = max_iter):

    # set current values
    current_solution = prs()
    current_residue = prepartitioned_residue(arr, current_solution)
    best_solution = current_solution

    # go through iterations - 1 (because did the first iteration above)
    for i in range(iterations - 1):

        # if new solution is better than old one, update
        current_solution = prs()
        new_residue = prepartitioned_residue(arr, current_solution)
        if new_residue < current_residue:
            best_solution = current_solution
            current_residue = new_residue

    # return solution array as well as final residue
    return best_solution, current_residue


# Hill Climbing Standard Representation
def hCsr(arr, iterations = max_iter):

    # set current values
    current_solution = srs()
    current_residue = standard_residue(arr, current_solution)
    best_solution = current_solution

    # go through iterations - 1 (because did the first iteration above)
    for i in range(iterations - 1):

        # if new solution (a neighbor this time) is better than old one, update
        current_solution = standard_neighbor(current_solution)
        new_residue = standard_residue(arr, current_solution)
        if new_residue < current_residue:
            best_solution = current_solution
            current_residue = new_residue

    # return solution array as well as final residue
    return best_solution, current_residue


# Hill Climbing Prepartitioned Representation
def hCpr(arr, iterations = max_iter):

    # set current values
    current_solution = prs()
    current_residue = prepartitioned_residue(arr, current_solution)
    best_solution = current_solution

    # go through iterations - 1 (because did the first iteration above)
    for i in range(iterations - 1):

        # if new solution (a neighbor this time) is better than old one, update
        current_solution = prepartitioned_neighbor(current_solution)
        new_residue = prepartitioned_residue(arr, current_solution)
        if new_residue < current_residue:
            best_solution = current_solution
            current_residue = new_residue

    # return solution array as well as final residue
    return best_solution, current_residue


# Simulated Annealing Standard Representation
def sAsr(arr, iterations = max_iter):

    # set current values
    current_solution = srs()
    current_residue = standard_residue(arr, current_solution)
    best_solution = current_solution

    # comment
    final = current_solution
    final_residue = current_residue

    # go through iterations - 1 (because did the first iteration above)
    for i in range(iterations - 1):

        # if new solution is better than old one, update
        current_solution = standard_neighbor(current_solution)
        new_residue = standard_residue(arr, current_solution)
        if new_residue < current_residue:
            best_solution = current_solution
            current_residue = new_residue

        # comment bruh
        else:
            cooling_factor = cooling_schedule(i)
            probability = math.exp(-(new_residue - current_residue)/cooling_factor)

            # comment bruh
            if random.random() <= probability:
                best_solution = current_solution
                current_residue = new_residue

        # comment bruh
        final_residue = standard_residue(arr, final)
        if current_residue < final_residue:
            final = current_solution

            # not really needed but here for clarity anyways
            final_residue = current_residue

    # return solution array as well as final residue
    return final, final_residue


# Simulated Annealing Prepartitioned Representation
def sApr(arr, iterations = max_iter):

    # set current values
    current_solution = prs()
    current_residue = prepartitioned_residue(arr, current_solution)
    best_solution = current_solution

    # comment
    final = current_solution
    final_residue = current_residue

    # go through iterations - 1 (because did the first iteration above)
    for i in range(iterations - 1):

        # if new solution is better than old one, update
        current_solution = prepartitioned_neighbor(current_solution)
        new_residue = prepartitioned_residue(arr, current_solution)
        if new_residue < current_residue:
            best_solution = current_solution
            current_residue = new_residue

        # comment bruh
        else:
            cooling_factor = cooling_schedule(i)

            probability = math.exp(-(new_residue - current_residue)/cooling_factor)

            # comment bruh
            if random.random() <= probability:
                best_solution = current_solution
                current_residue = new_residue

        # comment bruh
        final_residue = prepartitioned_residue(arr, final)
        if current_residue < final_residue:
            final = current_solution

            # not really needed but here for clarity anyways
            final_residue = current_residue

    # return solution array as well as final residue
    return final, final_residue


# Cooling Schedule for Simulated Annealing
def cooling_schedule(iter):
    return 10000000000 * (0.8)**math.floor(iter/300)


# Random Standard Representation Solution
def srs(iterations = fileSize):

    # create new solution array and iterate through to append elements
    solution = []
    for i in range(iterations):

        # 50% chance its 1, 50% chance its -1
        rand = random.randint(0, 1)
        if rand == 0:
            solution.append(-1)
        else:
            solution.append(1)

    return solution


# Random Prepartitioned Representation Solution
def prs(iterations = fileSize):

    # create new solution array and iterate through to append elements
    solution = []
    for i in range(iterations):

        # append random element
        rand = random.randint(1, fileSize)
        solution.append(rand)

    return solution


# calculates residue of a given standard representation on a given array
def standard_residue(arr, solution):
    assert(len(arr) == len(solution))
    length = len(arr)
    residue = 0
    for i in range(length):
        residue += arr[i]*solution[i]

    # return absolute value of residue, or distance to 0
    return abs(residue)


# calculates residue of a given prepartitioned representation on a given array
def prepartitioned_residue(arr, solution):
    assert(len(arr) == len(solution))
    length = len(arr)
    residue = 0
    new_arr = []
    indices_with_same_values = {}

    # create dictionary with keys as elements of solution array and values
    # as arrays with indices representing where and how many times
    # the elements show up
    for i in range(length):
        if solution[i] in indices_with_same_values:
            indices_with_same_values[solution[i]].append(i)
        else:
            indices_with_same_values[solution[i]] = [i]

    # for values in the dictionary, go through the original array and sum the
    # elements accordingly
    for value in indices_with_same_values:
        sum = 0
        for bruh in indices_with_same_values[value]:
            sum += arr[bruh]
        new_arr.append(sum)

    # make new array the same length as original array
    while len(new_arr) != length:
        new_arr.append(0)

    # run Karmarkar-Karp on the new array
    residue = kK(new_arr)
    return residue


# calculates a random neighbor given a standard represenation solution
def standard_neighbor(sr):

    # find 2 random numbers in the set
    rand1 = random.randint(0, fileSize-1)
    rand2 = random.randint(0, fileSize-1)

    # ensure rand 1 and rand 2 are not equal
    while rand2 == rand1:
        rand2 = random.randint(0, fileSize-1)

    # move element from set A1 to set A2
    sr[rand1] = -sr[rand1]

    # with probability 1/2 move an element from A2 to A1
    rand3 = random.randint(0, 1)
    if rand3 == 1:
        sr[rand2] = -sr[rand2]

    # return this new neighbor
    return sr


# calculates a random neighbor given a prepartitioned solution
def prepartitioned_neighbor(pr):

    # find 2 random numbers in the set
    rand1 = random.randint(0, fileSize-1)
    rand2 = random.randint(0, fileSize-1)

    # ensure p[1] and rand 2 are not equal
    while pr[rand1] == rand2:
        rand2 = random.randint(0, fileSize-1)

    # changing the partition one element lies in
    pr[rand1] = rand2

    # return this new neighbor
    return pr
