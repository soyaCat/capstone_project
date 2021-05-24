from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
import datetime
import time
import math
from collections import deque
import os

import random
import CustomFuncionFor_mlAgent as CF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

game = "sm4.exe"
env_path = "./build/" + game
save_picture_path = "./made_data/"
channel = EngineConfigurationChannel()

connection_test_count = 10

write_file_name_list_index_instead_of_correct_name = False
list_index_for_main = 0
list_index_for_goal = 1
generate_main = True
generate_goal = True


def slide_real_image(arr):
    list2 = list()
    list2.append(arr[0:40, 0:37, ])
    list2.append(arr[0:40, 38:92, ])
    list2.append(arr[0:40, 93:149, ])
    list2.append(arr[0:40, 150:204, ])
    list2.append(arr[0:40, 205:261, ])
    list2.append(arr[0:40, 262:300, ])

    list2.append(arr[41:95, 0:37, ])
    list2.append(arr[41:95, 38:92, ])
    list2.append(arr[41:95, 93:149, ])
    list2.append(arr[41:95, 150:204, ])
    list2.append(arr[41:95, 205:261, ])
    list2.append(arr[41:95, 262:300, ])

    list2.append(arr[96:151, 0:37, ])
    list2.append(arr[96:151, 38:92, ])
    list2.append(arr[96:151, 93:149, ])
    list2.append(arr[96:151, 150:204, ])
    list2.append(arr[96:151, 205:261, ])
    list2.append(arr[96:151, 262:300, ])

    list2.append(arr[152:207, 0:37, ])
    list2.append(arr[152:207, 38:92, ])
    list2.append(arr[152:207, 93:149, ])
    list2.append(arr[152:207, 150:204, ])
    list2.append(arr[152:207, 205:261, ])
    list2.append(arr[152:207, 262:300, ])

    list2.append(arr[208:250, 0:37, ])
    list2.append(arr[208:250, 38:92, ])
    list2.append(arr[208:250, 93:149, ])
    list2.append(arr[208:250, 150:204, ])
    list2.append(arr[208:250, 205:261, ])
    list2.append(arr[208:250, 262:300, ])
    return list2

def input_image():
    x = np.empty(shape = [50,50,3], dtype = int)
    x[:, :, :] = [255, 255, 255]
    list3 = []
    for i in range(30):
        list3.append(x.copy())
    return list3

def find_index_red(arr):
    minc = np.array([160, 80, 80])
    maxc = np.array([210, 95, 100])
    index_min = np.where(np.all(i >= minc, axis=2))
    index_max = np.where(np.all(i <= maxc, axis=2))

    intersects_0 = np.intersect1d(index_min[0], index_max[0])
    intersects_1 = np.intersect1d(index_min[1], index_max[1])

    data_collector = []
    col_list = []

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
    minc = np.array([79, 91, 250])
    maxc = np.array([82, 95, 255])
    index_min = np.where(np.all(i >= minc, axis=2))
    index_max = np.where(np.all(i <= maxc, axis=2))

    intersects_0 = np.intersect1d(index_min[0], index_max[0])
    intersects_1 = np.intersect1d(index_min[1], index_max[1])

    data_collector = []
    col_list = []

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
    minc = np.array([70, 165, 89])
    maxc = np.array([80, 190, 114])
    index_min = np.where(np.all(i >= minc, axis=2))
    index_max = np.where(np.all(i <= maxc, axis=2))

    intersects_0 = np.intersect1d(index_min[0], index_max[0])
    intersects_1 = np.intersect1d(index_min[1], index_max[1])

    data_collector = []
    col_list = []

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
    minc = np.array([160, 80, 90])
    maxc = np.array([210, 95, 100])
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
    minc = np.array([79, 91, 250])
    maxc = np.array([82, 95, 255])
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
    minc = np.array([70, 165, 89])
    maxc = np.array([80, 190, 114])
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

def final_process_image(arr):
    minc = np.array([0, 250, 0])
    maxc = np.array([0, 255, 0])
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
    arr[its_0-25:its_0+25,its_1-25:its_1+25,:] = [0,255,0]

    minc = np.array([0, 0, 250])
    maxc = np.array([0, 0, 255])
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
    arr[its_0 - 25:its_0 + 25, its_1 - 25:its_1 + 25, :] = [0, 0, 255]

    arr[200:250, 250:300, :] = [255,255,0]
    arr[0:50, 250:300, :] = [0,255,255]

    return arr
if __name__ == '__main__':
    for episodeCount in tqdm(range(connection_test_count)):
        wfnliiocn = write_file_name_list_index_instead_of_correct_name
        action = [1, 0, 0, 0, 0]
    x, y, w, h = [170, 195, 300, 250]
    vis_observation = cv2.imread("./made_data/0_main.jpg")
    vis_observation = cv2.cvtColor(vis_observation, cv2.COLOR_BGR2RGB)
    roi = vis_observation[y:y + h, x:x + w]
    roi = np.array(roi)
    arr = cv2.GaussianBlur(roi, (7, 7), 0)
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
            col_list = find_target_point_green(i, its_0, its_1)


        if cell != 'None':
            ansX = round(50 / np.shape(i)[1] * its_1[0]) # ansX 는 50x50에서 그 색의 값이 시작되는 x의 좌표
            ansY = round(50 / np.shape(i)[0] * its_0[0]) # ansY 는 50x50에서 그 색의 값이 시작되는 y의 좌표
            lenX = round((its_1[-1] - its_1[0]) / np.shape(i)[1] * 50)# ansX부터 lenX만큼 칠할것(범위)/
            lenY = round((its_0[-1] - its_0[0]) / np.shape(i)[0] * 50) # ansY부터 lenY만큼 칠할것(범위)

        if cell == 'obs':
            for j in range(50):
                for k in range(50):
                    list3[index][j][k] = [255, 0, 0]

        elif cell == 'target':
            for j in range(lenY):
                for k in range(lenX):
                    list3[index][ansY + j][ansX + k] = [0, 0, 255]

        elif cell == 'robot':
            for j in range(lenY):
                for k in range(lenX):
                    list3[index][ansY + j][ansX + k] = [0, 255, 0]

        col_list = find_target_point_red(i, its_0, its_1)

    result = np.empty(shape=[250, 300, 3], dtype = int)
    for i in range(30):
        result[(i//6)*50:(i//6)*50+50, (i%6)*50:(i%6)*50+50, :] = list3[i]
    plt.imshow(result)
    plt.show()
    result = final_process_image(result)
    plt.imshow(result)
    plt.show()
