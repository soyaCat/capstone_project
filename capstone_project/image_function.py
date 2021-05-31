import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

'''
real_red_minc = np.array([160, 80, 80])
real_red_maxc = np.array([210, 95, 100])
real_blue_minc = np.array([79, 91, 250])
real_blue_maxc = np.array([82, 95, 255])
real_green_minc = np.array([65, 165, 85])
real_green_maxc = np.array([85, 195, 117])
'''

real_red_minc = np.array([185, 30, 44])
real_red_maxc = np.array([220, 90, 110])
real_blue_minc = np.array([18, 76, 240])
real_blue_maxc = np.array([72, 130, 255])
real_green_minc = np.array([33, 130, 125])
real_green_maxc = np.array([60, 160, 150])

fake_red = np.array([161, 0, 0])
fake_orange = np.array([255, 75, 0])
fake_blue = np.array([0, 0, 185])
fake_green = np.array([0, 113, 15])
fake_yellow = np.array([192, 190, 0])
fake_pulple = np.array([108, 0, 97])

roi_parameter = [142,22,300,250]

count_y_green = 0
seta_H = 0
seta_W = 0


def slide_real_image(arr):
    list2 = list()
    list2.append(arr[0:47,0:40,])
    list2.append(arr[0:47,41:96,])
    list2.append(arr[0:47,97:154,])
    list2.append(arr[0:48,155:212,])
    list2.append(arr[0:51,213:267,])
    list2.append(arr[0:51,268:300,])

    list2.append(arr[48:95,0:37,])
    list2.append(arr[48:100,38:94,])
    list2.append(arr[48:100,95:152,])
    list2.append(arr[49:101,153:211,])
    list2.append(arr[52:101,212:267,])
    list2.append(arr[52:101,268:300,])

    list2.append(arr[96:147, 0:36, ])
    list2.append(arr[101:150, 37:93, ])
    list2.append(arr[101:152, 94:152, ])
    list2.append(arr[102:154, 153:212, ])
    list2.append(arr[102:150, 213:267, ])
    list2.append(arr[102:150, 268:300, ])

    list2.append(arr[151:195, 0:35, ])
    list2.append(arr[151:199, 36:92, ])
    list2.append(arr[152:202, 93:151, ])
    list2.append(arr[155:203, 152:209, ])
    list2.append(arr[156:203, 210:263, ])
    list2.append(arr[156:203, 264:300, ])

    list2.append(arr[196:250, 0:37, ])
    list2.append(arr[200:250, 38:94, ])
    list2.append(arr[203:250, 95:151, ])
    list2.append(arr[204:250, 152:207, ])
    list2.append(arr[204:250, 209:263, ])
    list2.append(arr[204:250, 262:300, ])
    return list2

def input_image():
    x = np.empty(shape = [50,50,3], dtype = int)
    x[:, :, :] = [255, 255, 255]
    list3 = []
    for i in range(30):
        list3.append(x.copy())
    return list3

def find_index_red(arr):
    minc = real_red_minc.copy()
    maxc = real_red_maxc.copy()
    index_min = np.where(np.all(arr >= minc, axis=2))
    index_max = np.where(np.all(arr <= maxc, axis=2))
    intersects_0 = np.intersect1d(index_min[0], index_max[0])
    intersects_1 = np.intersect1d(index_min[1], index_max[1])
    data_collector = []

    for k in intersects_0:
        for j in intersects_1:
            if np.all(arr[k][j]>minc) and np.all(arr[k][j]<maxc):
                data_collector.append(j)
    data_collector = set(data_collector)
    data_collector = list(data_collector)
    intersects_1 = np.array(data_collector)
    its_0 = list(intersects_0)
    its_1 = list(intersects_1)

    return its_0, its_1

def find_index_blue(arr):
    minc = real_blue_minc.copy()
    maxc = real_blue_maxc.copy()
    index_min = np.where(np.all(arr >= minc, axis=2))
    index_max = np.where(np.all(arr <= maxc, axis=2))
    intersects_0 = np.intersect1d(index_min[0], index_max[0])
    intersects_1 = np.intersect1d(index_min[1], index_max[1])
    data_collector = []

    for k in intersects_0:
        for j in intersects_1:
            if np.all(arr[k][j]>minc) and np.all(arr[k][j]<maxc):
                data_collector.append(j)
    data_collector = set(data_collector)
    data_collector = list(data_collector)
    intersects_1 = np.array(data_collector)
    its_0 = list(intersects_0)
    its_1 = list(intersects_1)

    return its_0, its_1

def find_index_green(arr):
    minc = real_green_minc.copy()
    maxc = real_green_maxc.copy()
    index_min = np.where(np.all(arr >= minc, axis=2))
    index_max = np.where(np.all(arr <= maxc, axis=2))
    intersects_0 = np.intersect1d(index_min[0], index_max[0])
    intersects_1 = np.intersect1d(index_min[1], index_max[1])
    data_collector = []
    for k in intersects_0:
        for j in intersects_1:
            if np.all(arr[k][j]>minc) and np.all(arr[k][j]<maxc):
                data_collector.append(j)
    data_collector = set(data_collector)
    data_collector = list(data_collector)
    intersects_1 = np.array(data_collector)

    its_0 = list(intersects_0)
    its_1 = list(intersects_1)

    return its_0, its_1

def find_target_point_red(arr, intersects_0, intersects_1):
    minc = real_red_minc.copy()
    maxc = real_red_maxc.copy()
    col_list = []
    for i in range(np.shape(arr)[0]):
        col_list.append(0)
    for i in intersects_0:
        if not np.any(np.where(4 == intersects_1)):
            for j in intersects_1:
                if np.all(arr[i][j] >= minc) and np.all(arr[i][j] <= maxc):
                    col_list[i] = j
                    break
        else:
            for j in np.flip(intersects_1):
                if np.all(arr[i][j] >= minc) and np.all(arr[i][j] <= maxc):
                    col_list[i] = j
                    break

    return col_list

def find_target_point_blue(arr, intersects_0, intersects_1):
    minc = real_blue_minc.copy()
    maxc = real_blue_maxc.copy()
    col_list = []

    for i in range(np.shape(arr)[0]):
        col_list.append(0)
    for i in intersects_0:
        if not np.any(np.where(4 == intersects_1)):
            for j in intersects_1:
                if np.all(arr[i][j] >= minc) and np.all(arr[i][j] <= maxc):
                    col_list[i] = j
                    break
        else:
            for j in np.flip(intersects_1):
                if np.all(arr[i][j] >= minc) and np.all(arr[i][j] <= maxc):
                    col_list[i] = j
                    break

    return col_list

def find_target_point_green(arr, intersects_0, intersects_1):
    minc = real_green_minc.copy()
    maxc = real_green_maxc.copy()
    global count_y_green
    global seta_H
    global seta_W
    col_list = []
    for i in range(np.shape(arr)[0]):
        col_list.append(0)
    for i in intersects_0:
        if not np.any(np.where(4 == intersects_1)):
            for j in intersects_1:
                if np.all(arr[i][j] >= minc) and np.all(arr[i][j] <= maxc):
                    col_list[i] = j
                    break
        else:
            for j in np.flip(intersects_1):
                if np.all(arr[i][j] >= minc) and np.all(arr[i][j] <= maxc):
                    col_list[i] = j
                    break

    count = 0
    end_index = 0
    for index, i in enumerate(col_list):
        if i != 0:
            count = count+1
            end_index = index

    if count > count_y_green:
        count_y_green = count
        seta_H = count_y_green-12
        seta_W = col_list[end_index-6]-col_list[end_index-count_y_green+6]

def final_process_image(arr):
    minc = fake_green
    maxc = fake_green
    index_min = np.where(np.all(arr >= minc, axis=2))
    index_max = np.where(np.all(arr <= maxc, axis=2))
    intersects_0 = np.intersect1d(index_min[0], index_max[0])
    intersects_1 = np.intersect1d(index_min[1], index_max[1])
    data_collector = []
    for k in intersects_0:
        for j in intersects_1:
            if np.all(arr[k][j]>=minc) and np.all(arr[k][j]<=maxc):
                data_collector.append(j)
    data_collector = set(data_collector)
    data_collector = list(data_collector)
    intersects_1 = np.array(data_collector)
    its_0 = round(np.mean(list(intersects_0)))
    its_1 = round(np.mean(list(intersects_1)))
    if(its_1<25):
        its_1 = 25
    elif(its_1>275):
        its_1 = 275
    if(its_0<25):
        its_0 = 25
    elif(its_1>275):
        its_1 = 275
    arr[its_0-25:its_0+25,its_1-25:its_1+25,:] = fake_green

    minc = fake_blue
    maxc = fake_blue
    index_min = np.where(np.all(arr >= minc, axis=2))
    index_max = np.where(np.all(arr <= maxc, axis=2))
    intersects_0 = np.intersect1d(index_min[0], index_max[0])
    intersects_1 = np.intersect1d(index_min[1], index_max[1])
    data_collector = []
    for k in intersects_0:
        for j in intersects_1:
            if np.all(arr[k][j] >= minc) and np.all(arr[k][j] <= maxc):
                data_collector.append(j)
    data_collector = set(data_collector)
    data_collector = list(data_collector)
    intersects_1 = np.array(data_collector)
    its_0 = round(np.mean(list(intersects_0)))
    its_1 = round(np.mean(list(intersects_1)))
    if (its_1 < 25):
        its_1 = 25
    elif (its_1 > 275):
        its_1 = 275
    if (its_0 < 25):
        its_0 = 25
    elif (its_1 > 275):
        its_1 = 275
    arr[its_0 - 25:its_0 + 25, its_1 - 25:its_1 + 25, :] = fake_blue

    arr[200:250, 250:300, :] = fake_yellow
    arr[0:50, 250:300, :] = fake_pulple

    return arr

def image_process(image):
    global count_y_green
    global seta_H
    global seta_W
    count_y_green = 0
    seta_H = 0
    seta_W = 0
    count_red_cell = 0
    red_cell_list = []
    x, y, w, h = roi_parameter
    seta = 0
    roi = image[y:y + h, x:x + w]
    roi = np.array(roi)
    arr = cv2.GaussianBlur(roi, (3, 3), 0)
    list2 = slide_real_image(arr)
    list3 = input_image()
    for index, i in enumerate(list2):
        cell = 'None'
        its_0, its_1 = find_index_red(i)
        if len(its_0) == 0 and len(its_1) == 0:
            its_0, its_1 = find_index_blue(i)
            if len(its_0) == 0 and len(its_1) == 0:
                its_0, its_1 = find_index_green(i)
                if len(its_0)!= 0 and len(its_1) != 0:
                    cell = 'robot'
            elif len(its_0)!= 0 and len(its_1) != 0:
                cell = 'target'
        elif len(its_0)!=0 and len(its_1) != 0:
            cell = 'obs'
        if cell == 'obs':
            col_list = find_target_point_red(i, its_0, its_1)
        if cell == 'target':
            col_list = find_target_point_blue(i, its_0, its_1)
        if cell == 'robot':
            find_target_point_green(i, its_0, its_1)

        if cell != 'None':
            ansX = round(50 / np.shape(i)[1] * its_1[0])# ansX 는 50x50에서 그 색의 값이 시작되는 x의 좌표
            ansY = round(50 / np.shape(i)[0] * its_0[0])# ansY 는 50x50에서 그 색의 값이 시작되는 y의 좌표
            lenX = round((its_1[-1] - its_1[0]) / np.shape(i)[1] * 50)# ansX부터 lenX만큼 칠할것(범위)
            lenY = round((its_0[-1] - its_0[0]) / np.shape(i)[0] * 50)# ansY부터 lenY만큼 칠할것(범위)
        if cell == 'obs':
            list3[index][:, :, :] = fake_red
            count_red_cell = count_red_cell + 1
            red_cell_list.append(index)
        elif cell == 'target':
            list3[index][ansY:ansY+lenY,ansX:ansX+lenX,:] = fake_blue
        elif cell == 'robot':
            list3[index][ansY:ansY+lenY,ansX:ansX+lenX,:] = fake_green

    if len(red_cell_list)>1:
        for i in red_cell_list:
            list3[i][:, :, :] = fake_orange

    result = np.empty(shape=[250, 300, 3], dtype = int)
    for i in range(30):
        result[(i//6)*50:(i//6)*50+50, (i%6)*50:(i%6)*50+50, :] = list3[i]
    result = final_process_image(result)

    seta = math.atan(seta_W/seta_H)
    seta = math.degrees(seta)


    return result, seta

