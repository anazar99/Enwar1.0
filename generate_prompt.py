import numpy as np

from camera_detect_im2text import image_to_text
from util import (calculate_initial_compass_bearing,
                  haversine,
                  load_and_preprocess_gps_coordinates,
                  load_file,
                  load_power_data,
                  read_extracted_lidar_data)


def generate_prompt(gps_information, lidar_detections, average_power, camera_details):
    camera_string = f'Scene details from camera: {camera_details} '
    lidar_string = f'LiDAR object detection results: {lidar_detections}'
    return (
        f"Given the following information captured from unit1, describe the physical and network environment this "
        f"information provides and be as detailed as possible. Do your best to estimate distances of objects and "
        f"their types, if there are any, and if there are any blockages, and how any of the provided information "
        f"affects the network if at all. Unit1 is a vehicle with a 360 camera (two 180 degree cameras), 360 degree 3D "
        f"lidar, and 4 60GHz receiver phased arrays and a GPS module, unit2 is a vehicle mounted with a GPS module "
        f"and a transmitter. "
        f"{gps_information} "
        f"{lidar_string}"
        f"{camera_string} "
        f"{average_power}")


def generate_prompt_sensing(gps_information, lidar_detections, camera_details):
    camera_string = f'Scene details from camera: {camera_details} '
    lidar_string = f'LiDAR object detection results: {lidar_detections}'
    return (
        f"Given the following information captured from unit1, describe the physical environment this information "
        f"provides and be as detailed as possible. Do your best to estimate distances of objects and their types, "
        f"if there are any, and if there are any blockages from obstacles, cars, pedestrians or anything that can "
        f"cause obstacles. Unit1 is a vehicle with a 360 camera (two 180 degree cameras), 360 degree 3D"
        f"lidar, and a GPS module, unit2 is a vehicle mounted with a GPS module and a transmitter. "
        f"{gps_information} "
        f"{lidar_string}"
        f"{camera_string} ")


if __name__ == "__main__":
    # used 0, 20, 103, 3302, 20630, 22431, 22477, 24799
    sample = 7993

    img_filetype = '.jpg'
    front_image_scenario_folder = 'data/scenario36/unit1/rgb5'
    unit1_front_image_load_files = load_file(front_image_scenario_folder, img_filetype)
    unit1_front_image_filepath = unit1_front_image_load_files[sample]
    print(unit1_front_image_filepath)

    back_image_scenario_folder = 'data/scenario36/unit1/rgb6'
    unit1_back_image_load_files = load_file(back_image_scenario_folder, img_filetype)
    unit1_back_image_filepath = unit1_back_image_load_files[sample]

    camera_details = image_to_text(unit1_front_image_filepath)
    camera_details += image_to_text(unit1_back_image_filepath)

    gps_filetype = 'txt'
    unit1_gps_scenario_folder = 'data/scenario36/unit1/gps1'
    unit1_gps_load_files = load_file(unit1_gps_scenario_folder, gps_filetype)
    unit1_gps_filepath = unit1_gps_load_files[sample]
    unit1_gps_data = load_and_preprocess_gps_coordinates(unit1_gps_filepath)

    unit2_gps_scenario_folder = 'data/scenario36/unit2/gps1'
    unit2_gps_load_files = load_file(unit2_gps_scenario_folder, gps_filetype)
    unit2_gps_filepath = unit2_gps_load_files[sample]
    unit2_gps_data = load_and_preprocess_gps_coordinates(unit2_gps_filepath)

    # Calculate distance and bearing
    distance = haversine(unit1_gps_data['latitude'], unit1_gps_data['longitude'], unit2_gps_data['latitude'],
                         unit2_gps_data['longitude'])
    bearing = calculate_initial_compass_bearing(unit1_gps_data['latitude'], unit1_gps_data['longitude'],
                                                unit2_gps_data['latitude'], unit2_gps_data['longitude'])

    gps_information = (f"The transmitter (Unit 2) is {distance} meters away at a bearing of {bearing} degrees from the "
                       f"receiver (Unit 1).")

    # Load power data from multiple folders
    pwr_scenario_folders = [
        'data/scenario36/unit1/pwr1',
        'data/scenario36/unit1/pwr2',
        'data/scenario36/unit1/pwr3',
        'data/scenario36/unit1/pwr4'
        ]

    N_BEAMS = 64
    # index 0 contains front receiver power, 1 right receiver, 2 back receiver, and 3 is the left receiver
    received_power = load_power_data(pwr_scenario_folders, sample, N_BEAMS)

    # Print power values for the first sample from each direction
    average_power = f"Average measured power from front receiver is {np.average(received_power[0])}, the right receiver is {np.average(received_power[1])}, the back receiver is {np.average(received_power[2])}, and the left receiver is {np.average(received_power[3])}"
    # print(power_string)

    lidar_detections_filename = f'dataset/lidar-detections/lidar-detections-sample-{sample}.txt'
    lidar_detections = read_extracted_lidar_data(lidar_detections_filename)

    # prompt = generate_prompt(gps_information, lidar_detections, average_power, camera_details)
    aware_dar_data_point_filename = f'dataset/aware-dar-dataset/sample{sample}.txt'

    prompt = generate_prompt_sensing(gps_information, lidar_detections, camera_details)
    print(prompt)
    # with open(aware_dar_data_point_filename, 'w') as file:
    #     file.write(prompt)

    # file.close()
