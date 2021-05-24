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
import cv2

game = "sm4.exe"
env_path = "./build_with_rot/" + game
save_picture_path = "./made_data/"
channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=1.0, target_frame_rate=60, capture_frame_rate=60)
env = UnityEnvironment(file_name=env_path, side_channels=[channel])
env.reset()
behavior_names = list(env.behavior_specs)
ConversionDataType = CF.ConversionDataType()
AgentsHelper = CF.AgentsHelper(env, string_log=None, ConversionDataType=ConversionDataType)

connection_test_count = 100

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
        action = [1, 0, 0, 0, 1]
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)
        env.step()

    for episodeCount in tqdm(range(10000)):
        behavior_name = behavior_names[0]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
        '''
        여기에 함수를 작성해 주세요
        함수 작성 요령
        genImg = Image_converter(vis_observation_list[0])
        vis_observation_list[0]를 카메라 입력으로 간주하고
        genImg를 생성 목표 이미지로 생각하시면 됩니다.
        생성되는 이미지의 사이즈는 64*64 픽셀로 부탁드립니다.
        생성목표가 되는 genImg는 made_data 폴더의 _goal.png파일을 참고하여 만들어주세요.
        -> 픽셀값이 필요하다면 말해주세요
        
        작성 함수
        1. 카메라 입력(vis_observation_list[0])에 대해 강화학습 및 로봇 조종용 genImg를 만드는 함수
        경기장 상단을 촬영하는 카메라의 이미지를 가지고 genImg 넘파이 어레이를 생성하는 함수를 만들어주세요
        genImg사이즈는 64*64*3입니다.
        위에서 말한 것과 같이 made_data 폴더의 _goal.png 파일을 참고하셔서 만드시면 됩니다.
        한 스텝을 돌 때마다 로봇이 조금씩 움직이는데 이에 따라 생성되는 genImg도 조금씩 바뀌여야 합니다.
        로봇이 어떻게 움직이는지 보고 싶으시다면 connection_test_count를 4000정도까지 증가시켜 주신 후 코드를 돌려주세요
        이미지를 만들 때마다 made_data 폴더 혹은 다른 폴더에 이미지를 저장하시면 됩니다.
        이미지 저장코드는 위의 예시에서와 같이
        save_numpy_file('_mine', list_index_for_main, wfnliiocn, episodeCount)을 사용하여 주세요
        저장 경로를 바꾸시고 싶으시다면
        save_numpy_file 함수 내에서 save_picture_path를 지정한 경로로 바꾸시면 됩니다.
        
        -> 요구사항이나 궁금한 점이 있다면 연락주세요
        '''
        action = [1, 0, 0, 0, 0]
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)
        env.step()
    env.close()



