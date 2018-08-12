import numpy as np
import matplotlib.pyplot as plt

def data2array(filename):
    pos = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
                pass
            pos.append(float(lines))
    pos = np.array(pos)
    return pos


filename3 = 'data1.txt'
filename4 = "data3.txt"
pos3 = data2array(filename3)
pos4 = data2array(filename4)
x = np.arange(len(pos3))

plt.figure()
l1, = plt.plot(x, pos3)
l2, = plt.plot(x, pos4, color = 'green', linestyle = '-.')
# l3, = plt.plot(x, pos4, color = 'red', linestyle = '--')
plt.xlim(0, 200)
plt.ylim(0.75, 1)
# plt.plot(x,pos3,pos4)
plt.legend(handles = [l1, l2,], labels = ['model 3', 'model 4'], loc = 'upper right')
plt.title("title")
plt.xlabel("Iterations", fontproperties = 'Times New Roman',fontsize = 15)
plt.ylabel("Fit Function", fontproperties = 'Times New Roman',fontsize = 15)
plt.grid(True) ##增加格点
plt.show()
