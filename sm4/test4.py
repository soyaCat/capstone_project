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
channel.set_configuration_parameters(time_scale=1.0, target_frame_rate=60, capture_frame_rate=60)
env = UnityEnvironment(file_name=env_path, side_channels=[channel])
env.reset()
behavior_names = list(env.behavior_specs)
ConversionDataType = CF.ConversionDataType()
AgentsHelper = CF.AgentsHelper(env, string_log=None, ConversionDataType=ConversionDataType)

connection_test_count = 10

write_file_name_list_index_instead_of_correct_name = False
list_index_for_main = 0
list_index_for_goal = 1
generate_main = True
generate_goal = True


def save_numpy_file(append_name, list_index, wfnliiocn, episodeCount):
    im = Image.fromarray(vis_observation_list[list_index].astype('uint8'), 'RGB')
    if wfnliiocn == False:
        im.save(save_picture_path + str(episodeCount) + append_name + '.jpg')
    else:
        im.save(save_picture_path + str(list_index) + '.jpg')


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
        behavior_name = behavior_names[0]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
        wfnliiocn = write_file_name_list_index_instead_of_correct_name
        if generate_main is True:
            save_numpy_file('_main', list_index_for_main, wfnliiocn,
                            episodeCount)  # made_data 폴더 내에 사진 파일이 저장되니 어떤 데이터를 다루는지 알고 싶다면 참고 부탁드립니다.
        if generate_goal is True:
            save_numpy_file('_goal', list_index_for_goal, wfnliiocn,
                            episodeCount)  # made_data 폴더 내에 목표 이미지 파일이 저장되니 참고 부탁드립니다.
        action = [1, 0, 0, 0, 0]
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)
        env.step()
    x, y, w, h = [170, 195, 300, 250]
    roi = vis_observation_list[0][y:y + h, x:x + w]

    arr = np.array(roi)
    list2 = slide_real_image(arr)

    for i in list2:
        print("이미지의 픽셀값들")
        (width,height,channel) = np.shape(i)
        print(width)
        print(height)
        print(channel)
        for w in range(width):
            print("{}줄의 픽셀".format(w))
            print(i[w])
        plt.imshow(i)
        plt.show()



    '''np.where()어떤조건에 있는 행렬의 인덱스값을 리턴
    인덱스 값 최고값과 최저값
    너비 알수 잇음, 높이도 알수 있음, 중앙값도 알수 있음'''

    for episodeCount in tqdm(range(10000)):
        behavior_name = behavior_names[0]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)

        action = [1, 0, 0, 0, 0]
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)
        env.step()
    env.close()