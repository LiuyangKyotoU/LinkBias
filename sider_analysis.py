import numpy as np

alpha = 0.5
ans = []
for time in range(5):
    with open('results/sider_test/sider_' + str(alpha) + '_' + str(time) + '.txt', 'r') as f:
        lines = f.readlines()
    tmp = [float(i.split()[0]) for i in lines]
    ans.append(tmp)
ans = np.array(ans)
np.mean(ans, axis=0), np.std(ans, axis=0)


alpha = 2.3
ans = []
for time in range(5): #01239
    if time == 7:
        continue
    with open('results/sider_test/gap_sider_' + str(alpha) + '_' + str(time) + '.txt', 'r') as f:
        lines = f.readlines()
    tmp = [float(i.split()[0]) for i in lines]
    ans.append(tmp)
ans = np.array(ans)
np.mean(ans, axis=0), np.std(ans, axis=0)
