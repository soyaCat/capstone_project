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

game = "robot_moving_simu.exe"
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
generate_main = True


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
            save_numpy_file('_main', list_index_for_main, wfnliiocn,
                            episodeCount)  # made_data 폴더 내에 사진 파일이 저장되니 어떤 데이터를 다루는지 알고 싶다면 참고 부탁드립니다.
        action = [1, 0, 0, 0, 0]
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)
        env.step()

        # 여기부터@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    robot = [0, 0]  # 로봇의 초기 좌표 (임시값) (0,0 이면 왼쪽맨위)
    target = [7, 6]  # 화물의 초기 좌표 (임시값)
    goal = [11, 9]  # 목표지점의 좌표 (임시값)

    move_target = [1, 0, 0, 0]

    action = set_robot(vis_observation_list[0])
    action = move_robot(vis_observation_list[0], move_target)

    while 1:
        behavior_name = behavior_names[0]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)

        m = 0
        n = 0

        if robot != target:  # 로봇의 좌표가 화물의 좌표와 일치하지 않으면 화물쪽으로 이동한다.
            vis_observation_list[0] = 1  # 움직임설정, 위, 아래, 뒤, 앞
            m = target[0] - robot[0]
            n = target[1] - robot[1]

            if m > 0:
                for i in range(m, 0, -1):
                    move_target = [0, 0, 0, 1]
            elif m < 0:
                for i in range(m, 0):
                    move_target = [0, 0, 1, 0]
            if n > 0:
                for i in range(n, 0, -1):
                    move_target = [0, 1, 0, 0]
            elif n < 0:
                for i in range(n, 0):
                    move_target = [1, 0, 0, 0]
        elif robot == target:
            vis_observation_list[0] = 0# ?
            break

            # 로봇을 움직이도록 명령을 보내주는 코드

    while 1:  # 화물의 위치에 도달한 후 다시 목표까지 이동한다.
        m = 0
        n = 0

        if robot != goal:
            vis_observation_list[0] = 1
            m = goal[0] - robot[0]
            n = goal[1] - robot[1]

            if m > 0:
                for i in range(m, 0, -1):
                    move_target = [0, 0, 0, 1]
            elif m < 0:
                for i in range(m, 0):
                    move_target = [0, 0, 1, 0]
            if n > 0:
                for i in range(n, 0, -1):
                    move_target = [0, 1, 0, 0]
            elif n < 0:
                for i in range(n, 0):
                    move_target = [1, 0, 0, 0]
        elif robot == goal:  # 운반을 완료하고 제자리로 돌아간다.
            k = target[0] + goal[0]
            l = target[1] + goal[1]

            if k > 0:
                for i in range(k, 0, -1):
                    move_target = [0, 0, 1, 0]
            elif k < 0:
                for i in range(k, 0):
                    move_target = [0, 0, 0, 1]
            if l > 0:
                for i in range(l, 0, -1):
                    move_target = [1, 0, 0, 0]
            elif l < 0:
                for i in range(l, 0):
                    move_target = [0, 1, 0, 0]

            break

        '''
        여기에 함수를 작성해 주세요
        함수 작성 요령
        action = set_robot(vis_observation_list[0])
        action = move_robot(vis_observation_list[0], move_target)
        action 변수는 리스트형 변수로 위의 예시와 같이 나오게 해주시면 됩니다.
        action = [1이면 움직임, 0이면 안 움직임(,
                    1이면 위로 조금 이동, 0이면 이동 안함,
                    1이면 아래로 조금 이동, 0이면 이동 안함,
                    1이면 뒤로 조금 이동, 0이면 이동 안함,
                    1이면 앞으로 조금 이동, 0이면 이동 안함]
        move_target는 다음과 같이 주어집니다
        move_target =[1이면 위로 한 칸,
                       1이면 아래로 한 칸,
                        1이면 뒤로 한 칸,
                        1이면 앞으로 한 칸]
        move_target이 [1,0,0,1] 이런 경우는 고려하지 않으셔도 됩니다.

        작성 함수
        1. 로봇을 가장 가까운 행렬칸으로 이동시키는 함수
         로봇의 위치를 가장 가까운 행렬칸으로 이동시켜주는 함수를 작성해주세요
         추후에 처음 로봇이 자리를 찾아가거나, 아님 프로그램적인 오류가 발생했을 때 로봇을 초기화 하는 것에 사용할 것입니다.
        -> 시작시 로봇을 랜덤한 곳에 위치시키고 싶다면 저에게 말해주세요


        2. 로봇을 move_target에 맞게 이동시키는 함수
        로봇의 현재 위치에서 move_target 변수에 맞게 이동시키는 함수를 작성해주세요
        예를 들어 move_target이 [0,1,0,0]으로 주어질 경우 로봇을 아래로 한 칸 이동시킨 후 move_target = [0,0,0,0]으로 초기화 시켜주시면 됩니다.
        -> 요청 사항이 있다면 말해주세요

        -> 두 프로그램을 동시에 while문 안에 넣으실 필요는 없습니다.
        -> 필요하다면 파이썬 파일을 복사하셔서 따로따로 작성해주셔도 되세요
        -> 목적이 종료되면 break문을 통해 while문을 탈출하게끔 만들어주세요
        '''
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)
        env.step()
    env.close()
