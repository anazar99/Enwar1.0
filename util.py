import os
import open3d as o3d
import torch
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted

# Change to your device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_file(load_filepath, filetype):
    # fetch files
    try:
        file = [f for f in os.listdir(load_filepath) if f.endswith(filetype)][1]
    except FileNotFoundError:
        raise FileNotFoundError(f'No {filetype} file inside {load_filepath}.')

    relative_paths = natsorted(os.listdir(load_filepath))
    load_files = [os.path.join(load_filepath, path) for path in relative_paths]

    return load_files


def load_and_preprocess_point_cloud(filepath):
    # Determine if the file is a CSV or PLY format
    csv_format = filepath.endswith('csv')

    if csv_format:
        # Load point cloud data from CSV file
        df = pd.read_csv(filepath)
        points = np.stack(
                (df[' X (mm)'].values, df[' Y (mm)'].values, df[' Z (mm)'].values)).T / 1000.0  # Convert mm to meters
    else:
        # Load point cloud data from PLY file
        pc = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pc.points)

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Apply voxel downsampling
    pcd = pcd.voxel_down_sample(voxel_size=0.25)

    # Remove statistical outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd


def read_extracted_lidar_data(filename):
    with open(filename, 'r') as file:
        lines = [line.strip('\n') + '. ' for line in file]

    return "".join(map(str, lines))


def load_and_preprocess_gps_coordinates(gps_file):
    # Read GPS data from the text file
    with open(gps_file, 'r') as file:
        lines = file.readlines()
        latitude = float(lines[0].strip())
        longitude = float(lines[1].strip())
        altitude = 354

    gps_data = pd.DataFrame({'latitude': [latitude], 'longitude': [longitude], 'altitude': [altitude]})
    return gps_data


def haversine(lat1, lon1, lat2, lon2):
    # Ensure lat/lon values are floats
    if isinstance(lat1, pd.Series):
        lat1 = float(lat1.iloc[0])
    if isinstance(lon1, pd.Series):
        lon1 = float(lon1.iloc[0])
    if isinstance(lat2, pd.Series):
        lat2 = float(lat2.iloc[0])
    if isinstance(lon2, pd.Series):
        lon2 = float(lon2.iloc[0])
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def calculate_initial_compass_bearing(lat1, lon1, lat2, lon2):
    # Ensure lat/lon values are floats
    if isinstance(lat1, pd.Series):
        lat1 = float(lat1.iloc[0])
    if isinstance(lon1, pd.Series):
        lon1 = float(lon1.iloc[0])
    if isinstance(lat2, pd.Series):
        lat2 = float(lat2.iloc[0])
    if isinstance(lon2, pd.Series):
        lon2 = float(lon2.iloc[0])
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    x = math.sin(delta_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - (math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda))

    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    compass_bearing = (bearing + 360) % 360

    return compass_bearing


def load_power_data(directory, sample_index, N_BEAMS=64):
    pwrs_list = []
    for i, folder in enumerate(directory):
        pwr_files = natsorted(os.listdir(folder))  # Ensure files are sorted
        pwrs_list.append([os.path.join(folder, pwr_file) for pwr_file in pwr_files])

    # Select the sample_index from the loaded data
    selected_pwrs_array = np.zeros((len(directory), N_BEAMS))
    for i, pwr_files in enumerate(pwrs_list):
        if sample_index < len(pwr_files):
            selected_pwrs_array[i] = np.loadtxt(pwr_files[sample_index])
        else:
            raise IndexError(f'Sample index {sample_index} out of range for folder {directory[i]}')

    return selected_pwrs_array


def point_cloud_to_tensor_in_chunks(pcd, chunk_size=1000):
    points = np.asarray(pcd.points)
    # Chunk size for processing
    chunk_size_tensor = chunk_size  # Adjust this value based on your dataset size and performance requirements
    total_points = points.shape[0]
    points_tensor = torch.empty((1, 3, total_points), dtype=torch.float32, device=device)

    print("Converting and transferring points_tensor...")

    with tqdm(total=total_points, desc="Processing Points") as pbar:
        for start_idx in range(0, total_points, chunk_size_tensor):
            end_idx = min(start_idx + chunk_size_tensor, total_points)
            chunk = points[start_idx:end_idx].T
            chunk_tensor = torch.tensor(chunk).unsqueeze(0).float().to(device)
            points_tensor[:, :, start_idx:end_idx] = chunk_tensor
            pbar.update(end_idx - start_idx)

    print("Completed conversion and transfer.")

    return points_tensor


def extract_features_with_progress(points_tensor, model):
    with tqdm(total=points_tensor.size(2), desc="Extracting Features", leave=False) as pbar:
        for i in range(points_tensor.size(2)):
            feature = model(points_tensor[:, :, i:i + 1])
            pbar.update(1)
    return feature

