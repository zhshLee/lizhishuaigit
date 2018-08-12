import cv2
import numpy as np
import os
from functools import reduce

class Tracker(object):
    def __init__(self,tracker_type = "BOOSTING",draw_coord = True):
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        self.tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.tracker_type = tracker_type
        self.isWorking = False
        self.draw_coord = draw_coord
        #构造追踪器
        if int(minor_ver) < 3:
            self.tracker = cv2.Tracker_create(tracker_type)
        else:
            self.tracker = cv2.TrackerMIL_create()

    def initWorking(self,frame,box):

        if not self.tracker:
            raise Exception("追踪器未初始化")
        status = self.tracker.init(frame,box)
        if not status:
            raise Exception("追踪器工作初始化失败")
        self.coord = box
        self.isWorking = True

    def track(self,frame):

        message = None
        if self.isWorking:
            status,self.coord = self.tracker.update(frame)
            if True:
                message = {"coord":[((int(self.coord[0]), int(self.coord[1])),(int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3])))]}
                if self.draw_coord:
                    p1 = (int(self.coord[0]), int(self.coord[1]))
                    p2 = (int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3]))
                    message['msg'] = "is tracking"
        return frame,message


# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir = os.listdir(filepath)
    out = []
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        out.append(child) # .decode('gbk')是解决中文显示乱码问题
    return out


def str2float(s):
    def char2num(s):
        return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s]
    if s.find('.') == -1:
        return int(s)
    else:
        s = s.split('.')
        n = s[0] + s[1]  # 连接两个字符串,整体转换
        return reduce(lambda x, y: x * 10 + y, map(char2num, n)) / (10 ** len(s[1]))


def calcIOU(Reframe,GTframe):

    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1,y2+height2)
    starty = min(y1,y2)
    height = height1+height2-(endy-starty)

    if width <=0 or height <= 0:
        ratio = 0 # 重叠率为 0
    else:
        Area = width*height # 两矩形相交面积
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
    # return IOU
    return ratio

def paint(mat,x1,y1,x2,y2):
    cv2.rectangle(mat,(int(x1),int(y1)),(int(x1)+int(x2),int(y1)+int(y2)),(0,0,255),2)

# 读取groundtruth第一帧
gths = "groundtruth"
dataDir = 'C:\\Users\\LZS\\Desktop\\vot2016\\'
# dataType = eachFile(dataDir)
dataType = dataDir+'/birds1'
# for gthdir in dataType:
a = 'MIL'
tracker = Tracker(tracker_type=a)
txtFile = '{}/{}.txt'.format(dataType, gths)
f = open(txtFile,'r')  # 返回一个文件对象
sourceInLines = f.readlines()  # 按行读出文件内容
f.close()
# new = []  # 定义一个空列表，用来存储结果
new = np.loadtxt(os.path.join(dataDir, dataType, "groundtruth.txt"), delimiter=",")
spl = ','.join(sourceInLines[0].split(',')).strip('\n').split(',')
# for x in spl:
#     x = str2float(x)
#     new.append(x)

x_mins = np.min(new[:, [0, 2, 4, 6]], axis=1)
y_mins = np.min(new[:, [1, 3, 5, 7]], axis=1)
x_maxs = np.max(new[:, [0, 2, 4, 6]], axis=1)
y_maxs = np.max(new[:, [1, 3, 5, 7]], axis=1)

cl,dl = [int(x_mins[0]),int(y_mins[0])],[int(x_maxs[0]),int(y_maxs[0])]
id = 2
# 获取第一帧
frame = cv2.imread(dataType+'/'+'00000001.jpg')
# 设置初始跟踪对象的窗口大小
c, r, w, h = cl[0],cl[1],dl[0]-cl[0],dl[1]-cl[1]
track_window = (c,r,w,h)
# cv2.rectangle(frame,(c,r),(c+w,r+h),255,2)
# cv2.imshow("frame",frame)
# cv2.waitKey(4000)
bbox = track_window
tracker.initWorking(frame, bbox)
IoU = []
while(True):
    frame = cv2.imread(dataType + '/' + '{:0>8}.jpg'.format(id))
    if frame is None:
        break
    _, item = tracker.track(frame)
    temp = np.array(item['coord'][0])
    cv2.rectangle(frame, (int(x_mins[id-1]), int(y_mins[id-1])), (int(x_maxs[id-1]), int(y_maxs[id-1])), (0, 255, 0), 1)
    cv2.rectangle(frame, (temp[0][0], temp[0][1]), (temp[1][0], temp[1][1]), (255, 0, 0), 1)
    cv2.imshow("track", frame)
    cv2.imwrite("C:\\Users\\LZS\\Desktop\\pycharmproject\\demo\\{}".format(str(id-1)), frame)

    # io.imsave(savename, image)

    # iou = calcIOU([515,236,602,270],[512,234,599,270])

    iou = calcIOU([int(x_mins[id-1]), int(y_mins[id-1]), int(x_maxs[id-1]), int(y_maxs[id-1])],[temp[0][0], temp[0][1], temp[1][0], temp[1][1]])
    IoU.append(iou)

    # 画出它的位置
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 点击视频窗口，按q键退出
        break
    id += 1
print(np.average(np.array(IoU)))
cv2.destroyAllWindows()