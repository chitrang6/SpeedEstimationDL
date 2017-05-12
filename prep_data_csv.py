import glob
import os
import csv

current_path = os.getcwd()


speed_file_path = current_path + "/data/train.txt"


with open(speed_file_path) as f:
    speed_per_frame = f.readlines()

speed_per_frame = [float(x) for x in speed_per_frame]

#print (speed_per_frame)

folder_path = current_path + "/data/training_data/*.jpg"

file_name = glob.glob(folder_path)


if len(speed_per_frame) == len(file_name):
    print("Both tehe files are having same length")
    with open('speed_and_image.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['filepath', 'time', 'speed'])

        for i in range(len(speed_per_frame)):
            filewriter.writerow([file_name[i], i * (1 / 20), speed_per_frame[i]])





#for file_t in file_name:
    #print (file_t)
