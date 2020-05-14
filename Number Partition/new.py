# addition to partition.py functions

# imports
from partition import *
import random
import sys
import math
import time
import statistics


# globals
random.seed(2)
fileSize = 100
max_num = 100000
max_iter = 250


# main
if __name__ == "__main__":

    # for autograder
    if int(sys.argv[1]) == 0:

        # grab the array of numbers from an input file as determined by arguments
        arr = []
        with open(sys.argv[3],'r') as file:
        	for line in file:
        	   arr.append(int(line))

        # autograder requirements
        if int(sys.argv[2]) == 0:
            print(str(kK(arr)))

        if int(sys.argv[2]) == 1:
            print(str(rRsr(arr)[1]))

        if int(sys.argv[2]) == 2:
            print(str(hCsr(arr)[1]))

        if int(sys.argv[2]) == 3:
            print(str(sAsr(arr)[1]))

        if int(sys.argv[2]) == 11:
            print(str(rRpr(arr)[1]))

        if int(sys.argv[2]) == 12:
            print(str(hCpr(arr)[1]))

        if int(sys.argv[2]) == 13:
            print(str(sApr(arr)[1]))

    else:

        # value arrays
        kk, rrsr, rrpr, hcsr, hcpr, sasr, sapr = [], [], [], [], [], [], []

        # time arrays
        kk_t, rrsr_t, rrpr_t, hcsr_t, hcpr_t, sasr_t =  [], [], [], [], [], []
        sapr_t = []

        # TESTING #
        for i in range(5):

            # change the random seed for each iteration
            random.seed(random.randint(1, max_iter))
            generateInputFile(maxx = max_num)

            # reload array of numbers
            arr = []
            with open(sys.argv[3],'r') as file:
            	for line in file:
            	   arr.append(int(line))

            # runs
            start = time.time()
            kk.append(kK(arr))
            end = time.time()
            kk_t.append(end - start)

            start = time.time()
            rrsr.append(rRsr(arr)[1])
            end = time.time()
            rrsr_t.append(end - start)

            start = time.time()
            rrpr.append(rRpr(arr)[1])
            end = time.time()
            rrpr_t.append(end - start)

            start = time.time()
            hcsr.append(hCsr(arr)[1])
            end = time.time()
            hcsr_t.append(end - start)

            start = time.time()
            hcpr.append(hCpr(arr)[1])
            end = time.time()
            hcpr_t.append(end - start)

            start = time.time()
            sasr.append(sAsr(arr)[1])
            end = time.time()
            sasr_t.append(end - start)

            start = time.time()
            sapr.append(sApr(arr)[1])
            end = time.time()
            sapr_t.append(end - start)

            print("finished iteration " + str(i + 1))

        # everything
        all = []
        all.append(kk)
        all.append(rrsr)
        all.append(rrpr)
        all.append(hcsr)
        all.append(hcpr)
        all.append(sasr)
        all.append(sapr)
        all.append(kk_t)
        all.append(rrsr_t)
        all.append(rrpr_t)
        all.append(hcsr_t)
        all.append(hcpr_t)
        all.append(sasr_t)
        all.append(sapr_t)

        # take averages
        averages = []
        for x in all:
            averages.append(statistics.mean(x))

        for i in range(len(averages)):
            averages[i] = str(round(averages[i], 3))

        # doesnt work
        print("---------------------VALUES-------------------")
        print("----------------------------------------------")
        print(" KK | RRSR | RRPR | HCSR | HCPR | SASR | SAPR ")
        print("" + averages[0] + " | " + averages[1] + " | " + averages[2] + " | " +
            averages[3] + " | " + averages[4] + " | " + averages[5] + " | " +
            averages[6])
