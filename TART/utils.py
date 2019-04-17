#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: utils.py
"""

import os
import shutil
import sys
import xmltodict
from subprocess import check_output
import psutil
import platform


def get_os():
    return platform.system()

def get_used_memory():  # GB
    p = psutil.Process(os.getpid())
    used_memory = keep_xiaoshu(p.memory_info().rss / 1024 / 1024 / 1024, 3)
    return used_memory


def get_total_memory():  # GB
    total_memory = keep_xiaoshu(psutil.virtual_memory().total / (1024.0 * 1024.0 * 1024.0), 3)
    return total_memory


def get_free_memory():  # GB
    free_memory = keep_xiaoshu(psutil.virtual_memory().available / (1024.0 * 1024.0 * 1024.0), 3)
    return free_memory


def get_pid():
    return os.getpid()

def get_cpu():
    import psutil
    return psutil.cpu_count()


def get_proper_gpu(gpu_number, minimum_memory_per_gpu):
    if get_os() == 'Windows':  # can't use GPU on Windows
        return [0]

    assert gpu_number > 0
    assert minimum_memory_per_gpu > 0

    # from tensorflow.python.client import device_lib
    # local_device_protos = device_lib.list_local_devices()
    # find_gpu = False
    # for x in local_device_protos:
    #     if x.device_type == 'GPU':
    #         find_gpu = True
    #         break
    # if not find_gpu:
    #     return None

    xml = check_output('nvidia-smi -x -q', shell=True, timeout=30)
    json = xmltodict.parse(xml)
    gpus = json['nvidia_smi_log']['gpu']
    if not type(gpus) == list:
        gpus = [gpus]

    gpu_indexes = []
    for gpu in gpus:

        minor_number = gpu['minor_number']
        fb_memory_usage = gpu['fb_memory_usage']

        free_memory = int(fb_memory_usage['free'].split(' ')[0])

        if free_memory >= minimum_memory_per_gpu:
            gpu_indexes.append(int(minor_number))

    if len(gpu_indexes) < gpu_number:
        print('only {} gpus are available'.format(len(gpu_indexes)))
        return None
    return gpu_indexes[:gpu_number]


def set_gpus(gpu_index):
    if get_os() == 'Windows':  # can't use GPU on Windows
        return

    if type(gpu_index) == list:
        gpu_index = ','.join(str(_) for _ in gpu_index)
    if type(gpu_index) == int:
        gpu_index = str(gpu_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

def keep_xiaoshu(value,xiaoshuhou):
    tt = pow(10, xiaoshuhou)
    return int(value*tt)/float(tt)

def del_dir_under(dirname):
    os.system('rm -f {}/*'.format(dirname))

def check_file(filename):
    return os.path.isfile(filename)


def check_dir(dirname):
    return os.path.isdir(dirname)


def del_dir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)


def del_create_dir(dirname):
    if check_dir(dirname):
        del_dir(dirname)
    create_dir(dirname)

def del_file(filename):
    if os.path.isfile(filename):
        os.system('rm ' + filename)

def create_dir(dirname):
    os.makedirs(dirname)  # 递归创建目录

def new_dir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)
    assert check_dir(dirname)

def get_all_files(input_dir, suffix=None):
    files = []
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and (suffix is None or os.path.splitext(file_path)[1] == suffix):
            files.append(file_path)

    return files

def get_all_files_recursive(dirs, suffix=None):
    files = []
    for dir in dirs:
        for t in os.walk(dir):
            t_dir, t_cdirs, t_files = t
            if len(t_files) is not 0:
                for file in t_files:
                    file_path = os.path.join(t_dir, file)
                    if suffix is None or os.path.splitext(file_path)[1] == suffix:
                        print(file_path)
                        files.append(file_path)
    return files

def copy_file(from_file, to_file):
    os.system('cp \"{}\" \"{}\"'.format(from_file, to_file))


def move_file(from_file, to_file):
    os.system('mv \"{}\" \"{}\"'.format(from_file, to_file))

def get_filename(filepath):
    filepath = filepath.strip()
    while filepath and filepath[-1] == '/':
        filepath = filepath[:-1]

    file_s = filepath.split('/')
    if '.' in file_s[-1]:
        filename = file_s[-1].split('.')[0]
    else:
        filename = file_s[-1]
    return filename

def merge_dirs(dirs, output_dir):
    files = []

    for dir_now in dirs:
        files +=get_all_files(dir_now)

    for file in files:
        filename = get_filename(file)
        copy_file(file,output_dir+filename+file.split['.'][-1])


def convert_houzhui(data_dir, from_houzhui, to_houzhui):
    files = get_all_files(data_dir, from_houzhui)
    print('get {} {} files'.format(len(files), from_houzhui))

    for file in files:
        filename = get_filename(file)
        move_file(file, data_dir + filename + to_houzhui)

def create_dirs_from_dict(base_dir,dict):
    if dict is None:
        return
    for key in dict:
        try:
            create_dir(base_dir + '/' + key + '/')  # 递归创建目录
        except:
            pass
        create_dirs_from_dict(base_dir + '/' + key, dict[key])


def convert(model_path, name):
    import numpy as np

    parameters = {}
    para = np.load(model_path, encoding='latin1').item()
    for p in para:
        for ii in range(len(para[p])):
            if ii == 0:
                parameters['{}/filter:0'.format(p)] = para[p][ii]
            if ii == 1:
                parameters['{}/biases:0'.format(p)] = para[p][ii]

    for p in parameters:
        print(p)

    import pickle

    with open('./weights/' + name + '.pkl', 'wb') as f:
        pickle.dump(parameters, f, pickle.HIGHEST_PROTOCOL)
