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

game = "sm3.exe"
env_path = "./build/" + game
save_picture_path = "./made_data/"
channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=1.0, target_frame_rate=60, capture_frame_rate=60)
env = UnityEnvironment(file_name=env_path, side_channels=[channel])
env.reset()
behavior_names = list(env.behavior_specs)
ConversionDataType = CF.ConversionDataType()
AgentsHelper = CF.AgentsHelper(env, string_log=None, ConversionDataType=ConversionDataType)

connection_test_count = 0  #

write_file_name_list_index_instead_of_correct_name = False
list_index_for_main = 0
generate_main = True
behavior_name = behavior_names[0]


def save_numpy_file(append_name, list_index, wfnliiocn, episodeCount):
    im = Image.fromarray(vis_observation_list[list_index].astype('uint8'), 'RGB')
    if wfnliiocn == False:
        im.save(save_picture_path + str(episodeCount) + append_name + '.jpg')
    else:
        im.save(save_picture_path + str(list_index) + '.jpg')

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
    # ??? ????????? ????????? ?????????
    # print("????????? ?????? ????????? ????????? ???")
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
        if not np.where(1 == intersects_1):
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

def find_robot(arr):
    minc = np.array([70, 168, 97])
    maxc = np.array([80, 178, 107])
    index_min = np.where(np.all(arr >= minc, axis=2))
    index_max = np.where(np.all(arr <= maxc, axis=2))
    # ??? ????????? ????????? ?????????
    # print("????????? ?????? ????????? ????????? ???")
    # print(set(index_min[0]), set(index_max[0]))
    # print(set(index_min[1]), set(index_max[1]))

    intersects_0 = np.intersect1d(index_min[0], index_max[0])
    intersects_1 = np.intersect1d(index_min[1], index_max[1])

    data_collector = []
    for k in intersects_0:
        for j in intersects_1:
            if np.all(arr[k][j] > minc) and np.all(arr[k][j] < maxc):
                data_collector.append(j)
    data_collector = set(data_collector)
    data_collector = list(data_collector)
    intersects_1 = np.array(data_collector)

    center_axis_0 = np.min(intersects_0) + (np.max(intersects_0)-np.min(intersects_0))/2
    center_axis_1 = np.min(intersects_1) + (np.max(intersects_1) - np.min(intersects_1))/2

    center_axis_0 = round(center_axis_0)
    center_axis_1 = round(center_axis_1)

    return intersects_0, intersects_1, center_axis_0, center_axis_1

def get_cell_center_point(arr):
    asix_0 = np.shape(arr)[0]/10
    asix_1 = np.shape(arr)[1]/12

    X_cell = []
    for i in range(30):
        new_list = [round(asix_0+2*asix_0*(i//6)), round(asix_1+2*asix_1*(i%6))]
        X_cell.append(new_list)

    return X_cell

def get_most_close_center_point(c_0, c_1, x_cell):
    # ?????? ????????? ??? ??????
    min_len = (c_0 - x_cell[0][0]) ** 2 + (c_1 - x_cell[0][1]) ** 2
    min_index = 0
    for index, i in enumerate(x_cell):
        len = (c_0 - i[0]) ** 2 + (c_1 - i[1]) ** 2
        if len < min_len:
            min_len = len
            min_index = index

    return min_index

def get_image_and_preprocess():
    behavior_name = behavior_names[0]
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
    x, y, w, h = [170, 195, 300, 250]
    roi = vis_observation_list[0][y:y + h, x:x + w]
    arr = np.array(roi)

    return arr

def move_robot(action):
    actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
    env.set_actions(behavior_name, actionTuple)
    env.step()


def move_target_index_point(target_index, x_cell):
    while(1):
        arr = get_image_and_preprocess()
        _, _, c_0, c_1 = find_robot(arr)
        move_axis0 = c_0-x_cell[target_index][0]
        move_axis1 = c_1-x_cell[target_index][1]
        Is_close_target = False

        if abs(move_axis0)<=2 and abs(move_axis1)<=2:
            Is_close_target = True
            return Is_close_target, target_index

        elif abs(move_axis0)>abs(move_axis1):
            if move_axis0<0:
                action = [1,0,1,0,0]
                move_robot(action)
            else:
                action = [1,1,0,0,0]
                move_robot(action)

        else:
            if move_axis1<0:
                action = [1,0,0,0,1]
                move_robot(action)
            else:
                action = [1,0,0,1,0]
                move_robot(action)

def get_new_target_index_from_move_target(move_target, target_index):
    if(move_target[0] == 1):
        target_index = target_index-6
    elif(move_target[1] == 1):
        target_index = target_index+6
    elif(move_target[2] == 1):
        target_index = target_index - 1
    elif(move_target[3] == 1):
        target_index = target_index +1

    move_target = [0,0,0,0]
    return move_target, target_index

def instance_stop():
    arr = get_image_and_preprocess()
    plt.imshow(arr)
    plt.show()

if __name__ == '__main__':
    for episodeCount in tqdm(range(connection_test_count)):
        behavior_name = behavior_names[0]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
        wfnliiocn = write_file_name_list_index_instead_of_correct_name
        if generate_main is True:
            save_numpy_file('_main', list_index_for_main, wfnliiocn,
                            episodeCount)  # made_data ?????? ?????? ?????? ????????? ???????????? ?????? ???????????? ???????????? ?????? ????????? ?????? ??????????????????.
        action = [1, 0, 1, 0, 0]
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)
        env.step()

    instance_stop()

    arr = get_image_and_preprocess()
    x_cell = get_cell_center_point(arr)
    _, _, c_0, c_1 = find_robot(arr)
    target_index = get_most_close_center_point(c_0, c_1, x_cell)
    Is_close_target, target_index = move_target_index_point(target_index, x_cell)
    instance_stop()

    move_target = [1, 0, 0, 0]
    move_target, target_index = get_new_target_index_from_move_target(move_target, target_index)
    move_target_index_point(target_index, x_cell)
    instance_stop()

    move_target = [0, 0, 1, 0]
    move_target, target_index = get_new_target_index_from_move_target(move_target, target_index)
    move_target_index_point(target_index, x_cell)
    instance_stop()

    move_target = [0, 1, 0, 0]
    move_target, target_index = get_new_target_index_from_move_target(move_target, target_index)
    move_target_index_point(target_index, x_cell)
    instance_stop()

    move_target = [0, 0, 0, 1]
    move_target, target_index = get_new_target_index_from_move_target(move_target, target_index)
    move_target_index_point(target_index, x_cell)
    instance_stop()

    env.close()


