import serial
import time
from driver import driver

car = driver()

#serial configuration
ser = serial.Serial("/dev/rfcomm0", 9600,timeout = 0.5)
ser.isOpen()
speed = 0
while True:
    msg = ser.read()
    print(msg)
    if(msg == "V"):
        speed = ser.read()
        ser.write(speed.encode())
        speed = 10*int(speed)
        print(speed)
    if(msg == "F"):
        car.set_speed(speed, speed)
        ser.write("Forward".encode())
    elif(msg == "B"):
        car.set_speed(-speed, -speed)
        ser.write("Back".encode())
    elif(msg == "L"):
        car.set_speed(speed+15, speed-15)
        ser.write("Left".encode())
    elif(msg == "R"):
        car.set_speed(speed-15, speed+15)
        ser.write("Right".encode())
    elif(msg == "S"):
        car.set_speed(0, 0)
        ser.write("Stop".encode())
    elif(msg == "F1"):
        car.set_speed(speed, speed)
        ser.write("Forward".encode())
    elif(msg == "B1"):
        car.set_speed(-speed, -speed)
        ser.write("Back".encode())
    elif(msg == "L1"):
        count = 0
        ser.write("Left".encode())
        while True:
            car.set_speed(speed+15, speed-15)
            count += 1
            if count >= 80:
                ser.write("Finished".encode())
                break
    elif(msg == "R1"):
        count = 0
        ser.write("Right".encode())
        while True:
            car.set_speed(speed-15, speed+15)
            count += 1
            if count >= 80:
                ser.write("Finished".encode())
                break
    elif(msg == "S1"):
        car.set_speed(0, 0)
        ser.write("Stop".encode())
        
        
        