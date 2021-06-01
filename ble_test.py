import asyncio
import random
from bleak import BleakClient
import time

address = "94:B9:7E:AC:86:1A"
read_write_charcteristic_uuid = "33abb9fe-193f-4d1e-8616-bd5865a35eac"
message = ""
services = 0
client = 0

def get_correct_vec():
    a = random.randint(0,6)
    time.sleep(0.5)
    try:
        a = input()
    except Exception as e:
        pass
    message = ""
    if a=='w':
        message = "w"
    elif a=='s':
        message = "s"
    elif a=='a':
        message = "a"
    elif a=='d':
        message = "d"
    elif a=='q':
        message = "q"
    elif a=='e':
        message = "e"
    elif a=='0':
        message = "0"
    elif a=='1':
        message = "1"
    elif a=='2':
        message = "2"
    elif a=='3':
        message = "3"
    else:
        message = "p"

    left = 0
    right = 0
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

async def send_message(services, message):
    for service in services:
        for characteristic in service.characteristics:
            if characteristic.uuid == read_write_charcteristic_uuid:
                # 데이터 쓰기
                if 'write' in characteristic.properties:
                    await client.write_gatt_char(characteristic, bytes(message.encode()))


if __name__=="__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_services(address))
    while 1:
        message = get_correct_vec()
        if message == 'p00':
            break
        else:
            loop.run_until_complete(send_message(services, message))
    loop.run_until_complete(disconnect())
    print("finish")
