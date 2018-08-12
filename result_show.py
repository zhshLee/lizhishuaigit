#!usr/bin/python
import numpy as np
import os
import matplotlib.pyplot as plt
result_path='E:/MDnet/results/'
result_name='L.txt'


path=os.path.join(result_path,result_name)
x = np.loadtxt(path, dtype=str, delimiter=",")
data = []
for i in range(x.shape[0]):
    data_temp = [float(item) for item in x[i,0:11]]
    data.append(data_temp)
data=np.array(data)
frames_num = len(data)
# print(np.shape(data))
# calculate the average fps

fps_ave = sum(data[:,9])/frames_num
accuracy_ave = sum(data[:,8])/frames_num
print('The average fps is {}'.format(fps_ave))
print('The average accuracy of {} is {}'.format('marching',accuracy_ave))

success = []
overlap = np.arange(0.1,1.01,0.01)

#overlap=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
for over in overlap:
    count=0
    for item in data:
        if(item[8] >= over):
            count=count+1
    success.append(count/frames_num)
plt.figure(figsize=[6,6])
plt.plot(overlap,success,color='r',linewidth=1)
plt.title('Success plot of OPE')
plt.ylabel('Sucess rate')
plt.xlabel('Overlap threshold')
plt.show()