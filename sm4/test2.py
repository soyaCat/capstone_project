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

def class_img(list): # 요소 판별
    cnt1=0
    cnt2=0
    cnt3=0
    cnt4=0
    cnt5=0
    for i in range(50):
        if list[i][0]>200 and list[i][1]>150 and list[i][1]<200 and list[i][2]>150 and list[i][2]<200:
            cnt1+=1
        elif list[i][0]>100 and list[i][0]<130 and list[i][1]>100 and list[i][1]<140 and list[i][2]<250:
            cnt2+=1
        elif list[i][0]>200 and list[i][0]<230 and list[i][1]>230 and list[i][2]>200 and list[i][2]<220:
            cnt3+=1
        elif list[i][0]>250 and list[i][1]>120 and list[i][1]<150 and list[i][2]>80 and list[i][2]<90:
            cnt4+=1
        elif list[i][0]>50 and list[i][0]<100 and list[i][1]>250 and list[i][2]>250:
            cnt5+=1
    if cnt1>5:
        return 'obs'
    elif cnt2>5:
        return 'target'
    elif cnt3>5:
        return 'robot'
    elif cnt4>5:
        return 'goal'
    elif cnt5>5:
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
            save_numpy_file('_main', list_index_for_main, wfnliiocn, episodeCount) # made_data 폴더 내에 사진 파일이 저장되니 어떤 데이터를 다루는지 알고 싶다면 참고 부탁드립니다.
        if generate_goal is True:
            save_numpy_file('_goal', list_index_for_goal, wfnliiocn, episodeCount) # made_data 폴더 내에 목표 이미지 파일이 저장되니 참고 부탁드립니다.
        action = [1, 0, 0, 0, 0]
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)
        env.step()
    x, y, w, h = [170, 195, 300, 250]
    roi = vis_observation_list[0][y:y + h, x:x + w]

    arr=np.array(roi)
    list1=list()


    for i in range(0, 250, 50):
        for j in range(0, 300, 50):
            list1.append(arr[i:i + 50, j:j + 50, :])
            plt.imshow(arr[i:i + 50, j:j + 50, :])
            plt.show()
    list1=np.array(list1)
    list1_avg=list()
    for i in range(30):
        list1_avg.append(np.mean(list1[i],axis=0))
    list1_avg = np.array(list1_avg)
    for i in range(30):
        print(class_img(list1_avg[i]))

    for episodeCount in tqdm(range(10000)):
        behavior_name = behavior_names[0]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)



        action = [1, 0, 0, 0, 0]
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)
        env.step()
    env.close()