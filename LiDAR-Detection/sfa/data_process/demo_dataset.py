"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for the KITTI dataset
"""

import sys
import os
from builtins import int
from glob import glob
import pandas as pd

import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset
import cv2
import torch

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_data_utils import get_filtered_lidar
from data_process.kitti_bev_utils import makeBEVMap
import config.kitti_config as cnf


class Demo_KittiDataset(Dataset):
    def __init__(self, configs):
        self.dataset_dir = os.path.join(configs.dataset_dir, configs.foldername, configs.foldername[:10],
                                        configs.foldername)
        self.input_size = configs.input_size
        self.hm_size = configs.hm_size

        self.num_classes = configs.num_classes
        self.max_objects = configs.max_objects

        self.image_dir = os.path.join("../../data/scenario36/unit1/rgb5/")
        self.lidar_dir = os.path.join("../../data/scenario36/unit1/lidar1")
        # self.calib_dir = os.path.join("../../data/scenario36/unit1/calib")

        # self.image_dir = os.path.join(self.dataset_dir, "image_02", "data")
        # self.lidar_dir = os.path.join(self.dataset_dir, "velodyne_points", "data")
        self.label_dir = os.path.join(self.dataset_dir, "label_2", "data")
        # self.sample_id_list = sorted(glob(os.path.join(self.lidar_dir, '*.bin')))
        # self.sample_id_list = sorted(glob(os.path.join(self.lidar_dir, '*.csv')))
        # self.sample_id_list = [float(os.path.basename(fn)[:-4]) for fn in self.sample_id_list]

        self.sample_id_list = [i for i in range(24800)]
        self.num_samples = len(self.sample_id_list)

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        pass

    def load_bevmap_front(self, index):
        """Load only image for the testing phase"""
        sample_id = int(self.sample_id_list[index])
        img_path, img_rgb = self.get_image(sample_id)
        lidarData = self.get_lidar(sample_id)
        front_lidar = get_filtered_lidar(lidarData, cnf.boundary)
        front_bevmap = makeBEVMap(front_lidar, cnf.boundary)
        front_bevmap = torch.from_numpy(front_bevmap)

        metadatas = {
            'img_path': img_path,
        }

        return metadatas, front_bevmap, img_rgb

    def load_bevmap_front_vs_back(self, index):
        """Load only image for the testing phase"""
        sample_id = int(self.sample_id_list[index])
        img_path, img_rgb = self.get_image2(sample_id)
        lidarData = self.get_lidar2(sample_id)

        front_lidar = get_filtered_lidar(lidarData, cnf.boundary)
        front_bevmap = makeBEVMap(front_lidar, cnf.boundary)
        front_bevmap = torch.from_numpy(front_bevmap)

        back_lidar = get_filtered_lidar(lidarData, cnf.boundary_back)
        back_bevmap = makeBEVMap(back_lidar, cnf.boundary_back)
        back_bevmap = torch.from_numpy(back_bevmap)

        metadatas = {
            'img_path': img_path,
        }

        return metadatas, front_bevmap, back_bevmap, img_rgb

    def load_file(self, load_filepath, filetype):
        try:
            file = [f for f in os.listdir(load_filepath) if f.endswith(filetype)][1]
            file_path = os.path.join(load_filepath, file)
        except FileNotFoundError:
            raise FileNotFoundError(f'No {filetype} file inside {load_filepath}.')

        relative_paths = natsorted(os.listdir(load_filepath))
        load_files = [os.path.join(load_filepath, path) for path in relative_paths]

        return load_files

    def load_and_preprocess_point_cloud(self, filepath):
        # Determine if the file is a CSV or PLY format
        csv_format = filepath.endswith('csv')

        if csv_format:
            # Load point cloud data from CSV file
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            points = np.stack(
                    (df['X (mm)'].values, df['Y (mm)'].values, df['Z (mm)'].values)).T / 1000.0  # Convert mm to meters

        else:
            # Load point cloud data from PLY file
            points = []
            with open(filepath, 'r') as f:
                is_header = True
                for line in f:
                    if is_header:
                        if line.startswith('end_header'):
                            is_header = False
                        continue
                    x, y, z = map(float, line.strip().split()[:3])
                    points.append([x, y, z])
            points = np.array(points)

        # Assuming the intensity is not available in the csv or ply file, setting it to 0
        intensity = np.zeros((points.shape[0], 1))

        # Combine points and intensity
        points_with_intensity = np.hstack((points, intensity))

        return points_with_intensity

    def get_image1(self, idx):
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(idx))
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        return img_path, img

    def get_image2(self, idx):
        # Load the list of image files using the load_file method
        image_files = self.load_file(self.image_dir, 'jpg')
        img_path = image_files[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        return img_path, img

    def get_lidar1(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '{:06d}.bin'.format(idx))
        # assert os.path.isfile(lidar_file)
        file = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        return file

    def get_lidar2(self, idx):
        lidar_load_files = self.load_file(self.lidar_dir, 'csv')
        lidar_filepath = lidar_load_files[idx]
        points_with_intensity = self.load_and_preprocess_point_cloud(filepath=lidar_filepath)
        return points_with_intensity
