from PIL import Image
import random
import math
import numpy as np

random.seed(10)#保证每次运行程序随机同样的数据
def generate_voronoi_diagram(width, height, num_cells):
    image = Image.new("RGB", (width, height))
    putpixel = image.putpixel
    imgx, imgy = image.size
    nx = []
    ny = []
    nr = []
    ng = []
    nb = []
    ###随机生成25个点########################
    for i in range(num_cells):
        nx.append(random.randrange(imgx))#随机25个点
        ny.append(random.randrange(imgy))
        nr.append(random.randrange(256))
        ng.append(random.randrange(256))
        nb.append(random.randrange(256))
    #####################################################

    def sort(test_data, dataset, k = 2):
        diff_dis_array = test_data - dataset  # 使用numpy的broadcasting
        dis_array = (np.add.reduce(diff_dis_array ** 2, axis=-1)) ** 0.5  # 求距离
        dis_array_index = np.argsort(dis_array)  # 升序距离的索引
        return dis_array_index[0:k]

    def addWord(theIndex, word, pagenumber):
        theIndex.setdefault(word, []).append(pagenumber)  # 存在就在基础上加入列表，不存在就新建个字典key
    ####################################################
    data = np.array([nx,ny]).T #这是前面随机生成的25个点转置以下，方便计算
    count = {}#建个字典，存此点与哪几个点相邻。是{'22': [0, 1, 2, 17], ...}这种形式，22这个点和0，1，2，17点相邻
    for y in range(imgy):#找图像中每个像素点在随机的25个点中的最近邻点
        for x in range(imgx):
            testdata = np.array([x,y])
            dist_index = sort(testdata, data)
            putpixel((x, y), (nr[dist_index[0]], ng[dist_index[0]], nb[dist_index[0]]))#填充像素
            addWord(count,str(dist_index[0]),dist_index[1])#添加字典元素
    for i in range(num_cells):
        count[str(i)] = list(set(count[str(i)]))#删除字典中的重复元素
    disnum = {}#存点对应所有的平均距离
    for j in range(num_cells):#j从第一个点开始遍历
        dis = []  # 列表保存求的最近临值，j每次遍历重置
        for id in count[str(j)]:
            dist = math.hypot(nx[id]-nx[j],ny[id]-ny[j])#math自带欧氏距离计算函数
            dis.append(dist)
        dis = np.array(dis)
        distave = np.average(dis)#求该点最近距离平均数
        addWord(disnum, str(j), distave)#添加到字典disnum中
    #新建的sortcnt字典重新排列count字典，只是为了方便查看############
    sortcnt = {}
    for j in range(num_cells):
        sortcnt[str(j)] = count[str(j)]
    #################################################################
    print(sortcnt)#打印此点与哪几个点相邻，{'0': [16, 17, 18, 22, 23], ...}这种形式，第0这个点和第16, 17, 18, 22, 23点相邻
    print(disnum)#打印点对应所有的平均距离{'0': [132.71915693589654]..}形式，第0个点平均距离是132.。。
    ################################################################################
    image.save("VoronoiDiagram.png", "PNG")#显示维诺图
    image.show()
###主程序
generate_voronoi_diagram(500, 500, 25)#500*500像素，随机25个点