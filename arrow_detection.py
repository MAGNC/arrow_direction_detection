import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time
import multiprocessing

class Handle:

    def __init__(self):
        self.thresh = 0.4
        self.template_left = self.get_outline('template_left.png')
        self.template_right = self.get_outline('template_right.png')
        self.template_up = self.get_outline('template_up.png')
        self.template_down = self.get_outline('template_down.png')
        self.target = self.get_outline(str(sys.argv[1]))
        self.target_origin = cv.imread(str(sys.argv[1]), cv.IMREAD_COLOR)#目的照片的三原色版
        # 拿到了模板和要匹配的图片

        self.template = [self.template_left, self.template_right, self.template_up, self.template_down]
        self.template_width = self.get_width_high(self.template)[0]
        self.template_high = self.get_width_high(self.template)[1]
        #分别获得了模板的长宽高

    def get_outline(self, location_of_picture):
        img = cv.imread(location_of_picture, cv.IMREAD_GRAYSCALE)
        after_canny = cv.Canny(img, 100, 200)
        # cv.imshow('{}after_canny'.format(location_of_picture), after_canny)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        return after_canny


    def template_matching(self, template, target, matching_method):
        res = cv.matchTemplate(template, target, matching_method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        return min_loc, res


    def display_matching(self, template_res, threshold, target, width, high):  # 此函数画出匹配的部分
        loc = np.where(template_res >= threshold)  # 返回矩阵中大于阈值的位置
        for pt in zip(*loc[::-1]):  # zip(*loc)将loc的值转化为列表，然后再转化为元组，也就是所有的匹配结果的位置
            target = cv.rectangle(target, pt, (pt[0] + width, pt[1] + high), (0, 0, 255), 2)
        target = cv.cvtColor(target, cv.COLOR_BGR2RGB)
        self.plot(target)


    def get_width_high(self, template_set):
        template_width = [0, 0, 0, 0]
        template_high = [0, 0, 0, 0]
        for i in range(len(self.template)):
            template_width[i], template_high[i] = self.template[i].shape[::-1]
        print(template_width, template_high)
        return template_width, template_high


    def plot(self, converted_picture):
        plt.imshow(converted_picture)
        plt.show()


if __name__ == '__main__':
    a = Handle()
    start = time.time()

    # 拿到模板的宽和高

    all_match_matrix = []  # 拿到目标对于每个模板的匹配矩阵
    for single in a.template:
        b = np.asarray(a.template_matching(single, a.target, cv.TM_CCOEFF_NORMED)[1])
        all_match_matrix.append(b)

    count = [0, 0, 0, 0]
    print("开始寻找最佳匹配")
    for every_matrix, index, i in tqdm.tqdm(
            zip(all_match_matrix, [0, 1, 2, 3], [1, 2, 3, 4])):  # 对每个矩阵中的每一个相似值都和阈值相比较，超过就记1，看哪个超过阈值最多，哪个就最匹配
        every_matrix = every_matrix.reshape(every_matrix.size, 1)
        print("正在看第{}个模板".format(i))
        for x in tqdm.tqdm(np.nditer(every_matrix)):
            # 阵变成一行的矩阵，便于用nditor方法
            # print(x)
            if abs(x) > a.thresh:
                count[index] += x

    # 找到最大相似度的模板
    # print(count)
    max_value = max(count)
    max_index = count.index(max_value)  # 找到最大值对应的被匹配的模板
    max_similar_template = a.template[max_index]
    # print(all_match_matrix[max_index])#打印每个像素的相似值矩阵
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

    min_loc = a.template_matching(max_similar_template, a.target, cv.TM_CCOEFF_NORMED)[0]
    res = a.template_matching(max_similar_template, a.target, cv.TM_CCOEFF_NORMED)[1]
    a.display_matching(res, a.thresh, a.target_origin, a.template_width[max_index], a.template_high[max_index])

    # 转换RGB，输出对应模板
    converted_ = cv.cvtColor(a.template[max_index], cv.COLOR_BGR2RGB)
    a.plot(converted_)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
