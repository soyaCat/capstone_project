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
import tensorflow as tf
import asyncio
from bleak import BleakClient
import image_function as IMG_F

game = "sm3.exe"
env_path = "./build/" + game
save_picture_path = "./made_data/"
address = "94:B9:7E:AC:86:1A"
read_write_charcteristic_uuid = "33abb9fe-193f-4d1e-8616-bd5865a35eac"
message = ""
services = 0
client = 0

connection_test_count = 0  #

write_file_name_list_index_instead_of_correct_name = False
list_index_for_main = 0
generate_main = True

state_size = [64, 64, 3]
action_size = 4
load_model = True
train_mode = False
batch_size = 32
mem_maxlen = 50000
discount_factor = 0.9
learning_rate = 0.00025
epsilon_init = 0.05
cap = cv2.VideoCapture(0)

# 모델 저장 및 불러오기 경로
load_path = "./suc2_DQN/25000_model/model"

# Model 클래스 -> 함성곱 신경망 정의 및 손실함수 설정, 네트워크 최적화 알고리즘 결정
class Model():
    def __init__(self, model_name):
        self.input = tf.placeholder(shape=[None, state_size[0], state_size[1], state_size[2]], dtype=tf.float32)
        self.input_normalize = (self.input - (255.0 / 2)) / (255.0 / 2)
        with tf.variable_scope(name_or_scope=model_name):
            self.conv1 = tf.layers.conv2d(inputs=self.input_normalize, filters=32, activation=tf.nn.relu, kernel_size=[8, 8], strides=[4, 4], padding="SAME")
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, activation=tf.nn.relu, kernel_size=[4, 4], strides=[2, 2], padding="SAME")
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, activation=tf.nn.relu, kernel_size=[3, 3], strides=[1, 1], padding="SAME")
            self.flat = tf.layers.flatten(self.conv3)
            self.fc1 = tf.layers.dense(self.flat, 512, activation=tf.nn.relu)
            self.Q_Out = tf.layers.dense(self.fc1, action_size, activation=None)
        self.predict = tf.argmax(self.Q_Out, 1)
        self.target_Q = tf.placeholder(shape=[None, action_size], dtype=tf.float32)
        self.loss = tf.losses.huber_loss(self.target_Q, self.Q_Out)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)

class DQNAgent():
    def __init__(self):
        self.model = Model("Q")
        self.target_model = Model("target")
        self.memory = deque(maxlen=mem_maxlen)
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.epsilon = epsilon_init
        self.Saver = tf.train.Saver()
        if load_model == True:
            self.Saver.restore(self.sess, load_path)
            print("load_success!")

    def get_action(self, state):
        if self.epsilon > np.random.rand():
            return np.random.randint(0, action_size)
        else:
            predict = self.sess.run(self.model.predict, feed_dict={self.model.input: state})
            return np.asscalar(predict)

def find_robot(arr):
    fake_green = np.array([0, 113, 15])
    minc = np.array([0, 110, 10])
    maxc = np.array([10, 115, 20])
    index_min = np.where(np.all(arr >= minc, axis=2))
    index_max = np.where(np.all(arr <= maxc, axis=2))
    # 두 행렬의 교집합 프린트
    # print("조건에 맞는 요소의 인덱스 값")
    # print(set(index_min[0]), set(index_max[0]))
    # print(set(index_min[1]), set(index_max[1]))
    try:
        intersects_0 = np.intersect1d(index_min[0], index_max[0])
        intersects_1 = np.intersect1d(index_min[1], index_max[1])
    except Exception as e:
        pass

    data_collector = []
    for k in intersects_0:
        for j in intersects_1:
            if np.all(arr[k][j] >= minc) and np.all(arr[k][j] <= maxc):
                data_collector.append(j)
    data_collector = set(data_collector)
    data_collector = list(data_collector)
    intersects_1 = np.array(data_collector)
    try:
        center_axis_0 = np.min(intersects_0) + (np.max(intersects_0)-np.min(intersects_0))/2
        center_axis_1 = np.min(intersects_1) + (np.max(intersects_1) - np.min(intersects_1))/2
    except Exception as e:
        print(intersects_0)
        print(intersects_1)
        plt.imshow(arr)
        plt.show()

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
    # 가장 가까운 셀 검색
    min_len = (c_0 - x_cell[0][0]) ** 2 + (c_1 - x_cell[0][1]) ** 2
    min_index = 0
    for index, i in enumerate(x_cell):
        len = (c_0 - i[0]) ** 2 + (c_1 - i[1]) ** 2
        if len < min_len:
            min_len = len
            min_index = index

    return min_index

def get_image_and_preprocess():
    #ret, frame = cap.read()
    ret, img = cap.read()
    img = cv2.resize(img, (410, 300), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_s, seta = IMG_F.image_process(img)
    arr = np.array(result_s)
    addictive = 0
    # print(seta)
    if seta>1:
        addictive = 1
    elif seta<-1:
        addictive = -1
    return arr, addictive

def move_robot(action, addictive):
    loop = asyncio.get_event_loop()
    message = get_correct_vec(action,addictive)
    loop.run_until_complete(send_message(message))

def move_target_index_point(target_index, x_cell):
    while(1):
        arr, addictive = get_image_and_preprocess()
        _, _, c_0, c_1 = find_robot(arr)
        move_axis0 = c_0-x_cell[target_index][0]
        move_axis1 = c_1-x_cell[target_index][1]
        Is_close_target = False

        if abs(move_axis0)<2 and abs(move_axis1)<2:
            Is_close_target = True
            return Is_close_target, target_index

        elif abs(move_axis0)>abs(move_axis1):
            if move_axis0<0:
                action = [1,0,1,0,0]
                move_robot(action, addictive)
            else:
                action = [1,1,0,0,0]
                move_robot(action, addictive)

        else:
            if move_axis1<0:
                action = [1,0,0,0,1]
                move_robot(action, addictive)
            else:
                action = [1,0,0,1,0]
                move_robot(action, addictive)

def get_new_target_index_from_move_target(move_target, target_index):
    if(move_target[0] == 1):
        if(target_index//6!=0):
            target_index = target_index-6
    elif(move_target[1] == 1):
        if(target_index//6 != 4):
            target_index = target_index+6
    elif(move_target[2] == 1):
        if(target_index % 6 != 0):
            target_index = target_index - 1
    elif(move_target[3] == 1):
        if(target_index % 6 != 5):
            target_index = target_index +1

    move_target = [0,0,0,0]
    return move_target, target_index

def get_DQN(arr, agent):
    arr = np.array(arr, dtype='uint8')
    resize_arr = cv2.resize(arr, (60, 50), interpolation = cv2.INTER_LANCZOS4)
    result = np.empty(shape = [64, 64, 3])
    result[:,:] = [0,0,0]
    result[7:57,2:62,:] = resize_arr.copy()/255
    result = np.array(result*255, dtype=int)
    result = np.reshape(result ,(1, 64, 64, 3))
    action = agent.get_action(result)
    print("action:", action)#(0위, 1:아래, 2: 뒤, 3:앞)
    move_target = [0, 0, 0, 0]
    move_target[action] = 1
    return move_target

def get_correct_vec(action, addictive):
    message = ""
    if action[1] == 1:
        message = "a"
    elif action[2] == 1:
        message = "d"
    elif action[3] == 1:
        message = "s"
    elif action[4] == 1:
        message = "w"
    else:
        message = 'p'
    left = 0
    right = 0
    if addictive == 1:
        left = 9
    if addictive == -1:
        right = 9
    message = message+str(left)+str(right)
    return message

async def get_services(address):
    global services
    global client
    client = BleakClient(address)
    await client.connect()
    services = await client.get_services()
    print('connect')

async def disconnect():
    global client
    await client.disconnect()
    print('disconnect')

async def send_message(message):
    global services
    for service in services:
        for characteristic in service.characteristics:
            if characteristic.uuid == read_write_charcteristic_uuid:
                # 데이터 쓰기
                if 'write' in characteristic.properties:
                    await client.write_gatt_char(characteristic, bytes(message.encode()))


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_services(address))

    arr, addictive = get_image_and_preprocess()
    x_cell = get_cell_center_point(arr)
    _, _, c_0, c_1 = find_robot(arr)
    target_index = get_most_close_center_point(c_0, c_1, x_cell)
    Is_close_target, target_index = move_target_index_point(target_index, x_cell)
    agent = DQNAgent()
    move_target = [0, 0, 0, 0]

    while 1:
        move_target, target_index = get_new_target_index_from_move_target(move_target, target_index)
        print("target_index:", target_index)
        move_target_index_point(target_index, x_cell)
        arr, addictive = get_image_and_preprocess()
        move_target = get_DQN(arr, agent)

    loop.run_until_complete(disconnect())

