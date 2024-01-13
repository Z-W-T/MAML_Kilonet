import matplotlib.pyplot as plt
import numpy as np

file = open("/home/XiYang/MAML_KiloNet/source/logs/IbMr_random_44875/losses.txt")
content = file.read()

content = content[2:-2]
losses = content.split('], [')
l=[]
for loss in losses:
    nums = loss.split(',')
    nums = list(map(int,nums))
    l.append(nums)
l = np.array(l)
print(l.shape)

x = np.arange(len(l[0]))
plt.figure()
plt.plot(x,l[0])
plt.savefig('test.jpg')

