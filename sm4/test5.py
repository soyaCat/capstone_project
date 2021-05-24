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
env_path = "./build_with_rot/" + game
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

def find_index(arr):
    minc = np.array([185, 75, 85])
    maxc = np.array([215, 100, 105])
    index_min = np.where(np.all(arr >= minc, axis=2))
    index_max = np.where(np.all(arr <= maxc, axis=2))
    # 두 행렬의 교집합 프린트
    # print("조건에 맞는 요소의 인덱스 값")
    # print(set(index_min[0]), set(index_max[0]))
    # print(set(index_min[1]), set(index_max[1]))

    intersects_0 = np.intersect1d(index_min[0], index_max[0])
    # print(intersects_0)
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
    # print(intersects_1)

    return intersects_0, intersects_1


def find_target_point(arr, intersects_0, intersects_1):
    minc = np.array([185, 75, 85])
    maxc = np.array([215, 100, 105])
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

if __name__ == '__main__':
    for episodeCount in tqdm(range(connection_test_count)):
        wfnliiocn = write_file_name_list_index_instead_of_correct_name
        action = [1, 0, 0, 0, 0]
    x, y, w, h = [170, 195, 300, 250]
    vis_observation = cv2.imread("./made_data/0_main.jpg")
    vis_observation = cv2.cvtColor(vis_observation, cv2.COLOR_BGR2RGB)
    roi = vis_observation[y:y + h, x:x + w]
    arr = np.array(roi)
    list2 = slide_real_image(arr)

    for i in list2:
        its_0, its_1 = find_index(i)
        col_list = find_target_point(i, its_0, its_1)
        #print(its_0)
        #print(its_1)
        print(col_list)
        plt.imshow(i)
        plt.show()

