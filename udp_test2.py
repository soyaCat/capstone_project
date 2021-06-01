from m5stack import *
from m5ui import *
from uiflow import *
import socket

setScreenColor(0x111111)

localIP = "127.0.0.1"
localPort = 20001
bufferSize = 1024
msgFromServer = "Hello UDP Client"
bytesToSend = str.encode(msgFromServer)
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
UDPServerSocket.bind((localIP, localPort))
lcd.font(lcd.FONT_Ubuntu)
lcd.print('server is open', 0, 0, 0x000000)
lcd.print(localIP, 0, 15, 0x000000)
lcd.print(localPort, 0, 30, 0x000000)
while True:
    bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
    message = bytesAddressPair[0]
    address = bytesAddressPair[1]
    clientMsg = "Message from Client:{}".format(message)
    clientIP = "Client IP Address:{}".format(address)
    print(clientMsg)
    print(clientIP)