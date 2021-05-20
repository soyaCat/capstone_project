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
env_path = "./build_with_grid/" + game
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
    minc = np.array([190, 80, 90])
    maxc = np.array([210, 95, 100])
    index_min = np.where(np.all(i >= minc, axis=2))
    index_max = np.where(np.all(i <= maxc, axis=2))
    # 두 행렬의 교집합 프린트
    # print("조건에 맞는 요소의 인덱스 값")
    # print(set(index_min[0]), set(index_max[0]))
    # print(set(index_min[1]), set(index_max[1]))

    intersects_0 = np.intersect1d(index_min[0], index_max[0])
    print(intersects_0)
    intersects_1 = np.intersect1d(index_min[1], index_max[1])

    data_collector = []
    for k in intersects_0:
        for j in intersects_1:
            if np.all(arr[k][j]>minc) and np.all(arr[k][j]<maxc):
                data_collector.append(j)
    data_collector = set(data_collector)
    data_collector = list(data_collector)
    intersects_1 = np.array(data_collector)
    print(intersects_1)







def class_img(list):  # 요소 판별
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    for i in range(50):
        if list[i][0] > 200 and list[i][1] > 150 and list[i][1] < 200 and list[i][2] > 150 and list[i][2] < 200:
            cnt1 += 1
        elif list[i][0] > 100 and list[i][0] < 130 and list[i][1] > 100 and list[i][1] < 140 and list[i][2] < 250:
            cnt2 += 1
        elif list[i][0] > 200 and list[i][0] < 230 and list[i][1] > 230 and list[i][2] > 200 and list[i][2] < 220:
            cnt3 += 1
        elif list[i][0] > 250 and list[i][1] > 120 and list[i][1] < 150 and list[i][2] > 80 and list[i][2] < 90:
            cnt4 += 1
        elif list[i][0] > 50 and list[i][0] < 100 and list[i][1] > 250 and list[i][2] > 250:
            cnt5 += 1
    if cnt1 > 5:
        return 'obs'
    elif cnt2 > 5:
        return 'target'
    elif cnt3 > 5:
        return 'robot'
    elif cnt4 > 5:
        return 'goal'
    elif cnt5 > 5:
        return 'goal'
    else:
        return 'empty'

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
        find_index(i)
        plt.imshow(i)
        plt.show()
