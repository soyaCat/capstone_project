import asyncio
import random
from bleak import BleakClient
import time

address = "94:B9:7E:AC:86:1A"
read_write_charcteristic_uuid = "33abb9fe-193f-4d1e-8616-bd5865a35eac"
message = ""

def get_correct_vec():
    a = random.randint(0,6)
    time.sleep(0.5)
    try:
        a = int(input())
    except Exception as e:
        pass
    message = ""
    if a==0:
        message = "move_forward"
    elif a==1:
        message = "move_backward"
    elif a==2:
        message = "move_left"
    elif a==3:
        message = "move_right"
    elif a==4:
        message = "turn_left"
    elif a==5:
        message = "turn_right"
    else:
        message = "stop"

    print(message)
    return message

async def run(address, message):
    async with BleakClient(address) as client:
        services = await client.get_services()
        for service in services:
            for characteristic in service.characteristics:
                if characteristic.uuid == read_write_charcteristic_uuid:
                    # 데이터 쓰기
                    if 'write' in characteristic.properties:
                        await client.write_gatt_char(characteristic, bytes(message.encode()))
    print('disconnect')


if __name__=="__main__":
    loop = asyncio.get_event_loop()
    while 1:
        message = get_correct_vec()
        loop.run_until_complete(run(address, message))
    print("finish")
