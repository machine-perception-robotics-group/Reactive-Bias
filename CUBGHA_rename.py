import os
import glob
import shutil
import argparse
with open(f'./images.txt', 'r') as f:
    datalist = f.readlines()

paths = []
for i in datalist:
    path = i.split(' ')
    path = path[1].split('.')[:2]
    path = '.'.join(path)
    path = [path, 'jpg']
    path = '.'.join(path)
    paths.append(path)

os.makedirs(f'./GHA', exist_ok=True)

cls = sorted(os.listdir('./images/'))

for c in cls:
    os.makedirs(f'./GHA/{c}', exist_ok=True)
count = 1
for path in paths:
    shutil.copy(f'./CUB_GHA/{count}.jpg',f'./{path}')
    count+=1
