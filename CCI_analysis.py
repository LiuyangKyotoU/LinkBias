import numpy as np

unbiased = []
for alpha in [0.04, 0.02, 0.01, 0.005]:
    tmp2 = []
    for time in range(5):
        with open('results/CCI_test/unbiased_' + str(alpha) + '_' + str(time) + '.txt', 'r') as f:
            lines = f.readlines()
        tmp1 = []
        for line in lines[2:]:
            tmp1.append([float(i) for i in line.split()])
        tmp2.append(tmp1)
    unbiased.append(tmp2)
unbiased = np.array(unbiased)

unbiased = []
for alpha in [0.16]:
    tmp2 = []
    for time in range(5):
        # if time == 0: continue
        with open('results/CCI_test/unbiased_' + str(alpha) + '_' + str(time) + '.txt', 'r') as f:
            lines = f.readlines()
        tmp1 = []
        for line in lines[1:]:
            tmp1.append([float(i) for i in line.split()])
        tmp2.append(tmp1)
    unbiased.append(tmp2)
unbiased = np.array(unbiased)
print(np.mean(unbiased[0], axis=0)[:, 0])

unbiased = []
for alpha in [1.0, 0.1]:
    tmp2 = []
    for time in range(5):
        with open('results/CCI_test/pop_unbiased_' + str(alpha) + '_' + str(time) + '.txt', 'r') as f:
            lines = f.readlines()
        tmp1 = []
        for line in lines[1:]:
            tmp1.append([float(i) for i in line.split()])
        tmp2.append(tmp1)
    unbiased.append(tmp2)
unbiased = np.array(unbiased)

unbiased = []
for alpha in [2.0]:
    tmp2 = []
    for time in range(5): #23467
        with open('results/CCI_test/gap_' + str(alpha) + '_' + str(time) + '.txt', 'r') as f:
            lines = f.readlines()
        tmp1 = []
        for line in lines[1:]:
            tmp1.append([float(i) for i in line.split()])
        tmp2.append(tmp1)
    unbiased.append(tmp2)
unbiased = np.array(unbiased)
print(np.mean(unbiased[0], axis=0)[:, 0])
print(np.mean(unbiased[0], axis=0)[:, -5])
