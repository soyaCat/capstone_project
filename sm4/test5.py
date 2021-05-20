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
    list2.append(arr[1:49, 0:48, ])
    list2.append(arr[1:49, 49:99, ])
    list2.append(arr[1:49, 100:150, ])
    list2.append(arr[1:49, 151:200, ])
    list2.append(arr[1:49, 201:250, ])
    list2.append(arr[1:49, 251:300, ])

    list2.append(arr[50:100, 0:48, ])
    list2.append(arr[50:100, 49:99, ])
    list2.append(arr[50:100, 100:150, ])
    list2.append(arr[50:100, 151:200, ])
    list2.append(arr[50:100, 201:250, ])
    list2.append(arr[50:100, 251:300, ])

    list2.append(arr[101:151, 0:48, ])
    list2.append(arr[101:151, 49:99, ])
    list2.append(arr[101:151, 100:150, ])
    list2.append(arr[101:151, 151:200, ])
    list2.append(arr[101:151, 201:250, ])
    list2.append(arr[101:151, 251:300, ])

    list2.append(arr[152:199, 0:48, ])
    list2.append(arr[152:199, 49:99, ])
    list2.append(arr[152:199, 100:150, ])
    list2.append(arr[152:199, 151:200, ])
    list2.append(arr[152:199, 201:250, ])
    list2.append(arr[152:199, 251:300, ])

    list2.append(arr[200:250, 0:48, ])
    list2.append(arr[200:250, 49:99, ])
    list2.append(arr[200:250, 100:150, ])
    list2.append(arr[200:250, 151:200, ])
    list2.append(arr[200:250, 201:250, ])
    list2.append(arr[200:250, 251:300, ])
    return list2

def find_index(arr):
    minc = np.array([100, 100, 150])
    maxc = np.array([255, 200, 200])
    index_min = np.where(np.all(i >= minc, axis=2))
    index_max = np.where(np.all(i <= maxc, axis=2))
    # 두 행렬의 교집합 프린트
    print("조건에 맞는 요소의 인덱스 값")
    print(np.intersect1d(index_min[0], index_max[0]))
    print(np.intersect1d(index_min[1], index_max[1]))




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
        print("이미지의 픽셀값들")
        (width,height,channel) = np.shape(i)
        print(width)
        print(height)
        print(channel)
        find_index(i)
        plt.imshow(i)
        plt.show()
