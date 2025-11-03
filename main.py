import os
from langchain_community.vectorstores import FAISS
from RAG import (get_llm,
                 get_embeddings,
                 extract_docs,
                 get_vector_store,
                 get_conversation_chain,
                 get_conversation_chain_llama)


def main():
    # Initialize the HuggingFaceEndpoint
    print("Getting llm...")
    model_name = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
    llm_llama = get_llm(model_name)
    model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    llm_mistral = get_llm(model_name)

    print("Getting embedding model...")
    embeddings_model = "Alibaba-NLP/gte-large-en-v1.5"
    embeddings_chunk_size = 1024
    embeddings = get_embeddings(embeddings_model)

    # pdf_glob = "**/*pdf"
    # python_glob = "**/*py"
    txt_glob = "**/*txt"

    # ara_apis_and_code_dataset = 'dataset/ara-apis-and-code'
    # ara_dataset = "dataset/ara-gpt-dataset"
    # aware_dar_dataset = "dataset/aware-dar-dataset"
    environment_sensing_dataset = 'dataset/environment-sensing-dataset'
    # ericsson_dataset = "dataset/ericsson-dataset"
    print("Initialized embedding model.")

    print("Checking vector store...")
    vector_store_file_name = 'faiss-db/faiss_index'

    # this is something that lets me create a db everytime i add a new sample
    create_db_flag = 1

    if os.path.exists(vector_store_file_name) and create_db_flag == 1:
        print("Vector store exists")
        vector_store = FAISS.load_local(vector_store_file_name, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded")
    else:
        print("Vector store does not exist, creating vector store...")
        print("Extracting documents...")
        print("\tExtracting txts...")

        extracted_docs = extract_docs(environment_sensing_dataset, txt_glob, embeddings_chunk_size)
        # extracted_docs = extract_docs(ara_apis_and_code_dataset, python_glob, embeddings_chunk_size)
        # extracted_docs += extract_docs(ara_dataset, txt_glob, embeddings_chunk_size)
        # extracted_docs += extract_docs(aware_dar_dataset, txt_glob, embeddings_chunk_size)
        # extracted_docs += extract_docs(ericsson_dataset, pdf_glob, embeddings_chunk_size)

        vector_store = get_vector_store(extracted_docs, embeddings)
        vector_store.save_local(vector_store_file_name)
        print("Vector store saved and loaded")

    print("Creating conversation chain...")
    chain_mistral = get_conversation_chain(llm_mistral, vector_store)
    chain_llama = get_conversation_chain_llama(llm_llama, vector_store)
    print("Created conversation chain."
          "\nReady")

    # not busy, empty parking lot
    prompt_sample0 = 'Given the following information captured from unit1, describe the physical environment this information provides and be as detailed as possible. Do your best to estimate distances of objects and their types, if there are any, and if there are any blockages between unit1 and unit2 from obstacles, cars, pedestrians or anything that can cause obstacles. Unit1 is a vehicle with a 360 camera (two 180 degree cameras), 360 degree 3D lidar, and a GPS module, unit2 is a vehicle mounted with a GPS module and a transmitter. The transmitter (Unit 2) is 8.066753628120216 meters away at a bearing of 182.27478933305133 degrees from the receiver (Unit 1). LiDAR object detection results: A Vehicle is detected 15.07 meters away at a direction of 322.89 degrees. A Vehicle is detected 11.38 meters away at a direction of 229.71 degrees. A Vehicle is detected 9.83 meters away at a direction of 241.23 degrees. A Vehicle is detected 16.96 meters away at a direction of 181.47 degrees. Scene details from camera: The scene depicts a man wearing a red shirt standing in the middle of a city street near a parking lot. There are several cars parked in the parking lot, and some of them are visible in the background. The man appears to be gesturing with his hand, possibly indicating something to the driver of the car he is standing next to. In addition to the man and the parked cars, there are several trees scattered throughout the scene. One tree can be seen on the left side of the image, while another is located on the right side. There are also two benches present, one on the left side and another on the right side of the image. The scene is a bird''s-eye view of a parking lot from the perspective of a car. There are several vehicles parked in the lot, including a blue car, a white car, and a truck. Additionally, there are some trees visible in the background, providing a scenic setting for the parking lot. The sun is shining brightly, casting a warm and inviting glow over the area.'

    # not obvious blockage, car far away
    prompt_sample3302 = 'Given the following information captured from unit1, describe the physical environment this information provides and be as detailed as possible. Do your best to estimate distances of objects and their types, if there are any, and if there are any blockages between unit1 and unit2 from obstacles, cars, pedestrians or anything that can cause obstacles. Unit1 is a vehicle with a 360 camera (two 180 degree cameras), 360 degree 3D lidar, and a GPS module, unit2 is a vehicle mounted with a GPS module and a transmitter. The transmitter (Unit 2) is 134.9016963054195 meters away from the receiver (Unit 1) where Unit 1 is at a bearing of 178.86933709163418 degrees from Unit 2. LiDAR object detection results: A Vehicle is detected 15.51 meters away at a direction of 66.55 degrees. A Vehicle is detected 22.92 meters away at a direction of 29.51 degrees. A Vehicle is detected 18.97 meters away at a direction of 80.52 degrees. A Vehicle is detected 12.68 meters away at a direction of 201.62 degrees. A Vehicle is detected 3.70 meters away at a direction of 205.08 degrees. A Vehicle is detected 26.97 meters away at a direction of 116.24 degrees. Scene details from camera: The scene depicts a car driving down a city street, as seen from the driver''s point of view. The car is parked on the side of the road, and there are several other vehicles visible in the background. The sun is shining brightly in the sky, casting a warm and inviting glow over the scene. Additionally, there are trees lining the street, adding a natural touch to the urban environment.The scene depicts a view from the back of a car as it drives down a busy city street. There are multiple cars visible in the scene, with some parked on the side of the road and others passing by. In addition to the cars, there are several pedestrians walking along the sidewalks, adding to the bustling atmosphere of the city. Some of the pedestrians are closer to the camera, while others are further away. A traffic light can be seen in the distance, indicating the flow of traffic on the street.'

    #ambulance obvious blockage
    prompt_sample20630 = 'Given the following information captured from unit1, describe the physical environment this information provides and be as detailed as possible. Do your best to estimate distances of objects and their types, if there are any, and if there are any blockages between unit1 and unit2 from obstacles, cars, pedestrians or anything that can cause obstacles. Unit1 is a vehicle with a 360 camera (two 180 degree cameras), 360 degree 3D lidar, and a GPS module, unit2 is a vehicle mounted with a GPS module and a transmitter. The transmitter (Unit 2) is 112.79471527640717 meters away from the receiver (Unit 1) and Unit 1 is at a bearing of 359.8653702653738 degrees from Unit 2''s position. LiDAR object detection results: A Vehicle is detected 3.21 meters away at a direction of 24.82 degrees. A Vehicle is detected 15.41 meters away at a direction of 158.79 degrees. Scene details from camera: The scene depicts a highway with two vehicles, an ambulance and a car, traveling in opposite directions. The ambulance is driving on the left side of the road, while the car is driving on the right side of the road. The ambulance has a siren blaring, indicating that it is responding to an emergency call. In the background, there is a bridge spanning over the highway.The scene depicts a highway with an ambulance parked on the side of the road. The ambulance is visible from the back of the car, which is traveling in the same direction as the ambulance. In addition to the ambulance, there are several other vehicles on the road, including cars, trucks, and vans.'
    prompt = "Given the following information captured from unit1, describe the physical environment this information provides and be as detailed as possible. Do your best to estimate distances of objects and their types, if there are any, and if there are any blockages from obstacles, cars, pedestrians or anything that can cause obstacles between unit 1 and unit 2. Unit1 is a vehicle with a 360 camera (two 180 degree cameras), 360 degree 3D lidar, and a GPS module, unit2 is a vehicle mounted with a GPS module and a transmitter. The transmitter (Unit 2) is 7.7534236928011895 meters away at a bearing of 358.8162531212902 degrees from the receiver (Unit 1). LiDAR object detection results: A Vehicle is detected 16.77 meters away at a direction of 350.83 degrees. A Vehicle is detected 17.08 meters away at a direction of 335.32 degrees. A Vehicle is detected 16.64 meters away at a direction of 195.72 degrees. Scene details from camera: The scene depicts a busy city street filled with cars, trucks, and motorcycles. There are several vehicles on the road, including a silver car, a white car, a truck, and a motorcycle. Some of the vehicles are parked on the side of the street, while others are moving along the road. The sun is shining brightly in the sky, casting a warm and inviting glow over the entire scene.The scene is captured from the back of a car driving down a busy city street. There are several cars in front of the car, creating a congested traffic situation. Some of these cars are parked on the side of the street, while others are moving along with the flow of traffic. In addition to the cars, there are several trees and palm trees lining the street, adding to the scenic view."
    # busy intersection no blockage
    prompt_sample24799_text_gps_only = 'Given the following information captured from unit1, describe the physical environment this information provides and be as detailed as possible. Do your best to estimate distances of objects and their types, if there are any, and if there are any blockages between unit1 and unit2 from obstacles, cars, pedestrians or anything that can cause obstacles. Unit1 is a vehicle with a 360 camera (two 180 degree cameras), 360 degree 3D lidar, and a GPS module, unit2 is a vehicle mounted with a GPS module and a transmitter. The transmitter (Unit 2) is 10.539957539355683 meters away at a bearing of 159.9010115892885 degrees from the receiver (Unit 1).'
    prompt_sample24799_image_gps = 'Given the following information captured from unit1, describe the physical environment this information provides and be as detailed as possible. Do your best to estimate distances of objects and their types, if there are any, and if there are any blockages between unit1 and unit2 from obstacles, cars, pedestrians or anything that can cause obstacles. Unit1 is a vehicle with a 360 camera (two 180 degree cameras), 360 degree 3D lidar, and a GPS module, unit2 is a vehicle mounted with a GPS module and a transmitter. The transmitter (Unit 2) is 10.539957539355683 meters away at a bearing of 159.9010115892885 degrees from the receiver (Unit 1). Scene details from camera: The scene depicts a car driving underneath a bridge on a sunny day. There are several cars parked along the side of the road, and a traffic light can be seen in the distance. The camera is positioned from the front of the car, providing a bird''s-eye view of the surrounding area.In the scene, there are several cars driving on a busy city street underneath an overpass. The view is captured from the back of a car, providing a bird''s-eye view of the traffic below. There are multiple cars visible in the image, some of which are parked on the side of the road, while others are moving along the street. Traffic lights can be seen in the distance, indicating the flow of traffic and the need for drivers to follow the rules of the road. Overall, the scene captures the hustle and bustle of a busy city street, showcasing the diverse range of vehicles and pedestrians navigating their way through the urban environment.'
    prompt_sample24799_lidar_gps = 'Given the following information captured from unit1, describe the physical environment this information provides and be as detailed as possible. Do your best to estimate distances of objects and their types, if there are any, and if there are any blockages between unit1 and unit2 from obstacles, cars, pedestrians or anything that can cause obstacles. Unit1 is a vehicle with a 360 camera (two 180 degree cameras), 360 degree 3D lidar, and a GPS module, unit2 is a vehicle mounted with a GPS module and a transmitter. The transmitter (Unit 2) is 10.539957539355683 meters away at a bearing of 159.9010115892885 degrees from the receiver (Unit 1). LiDAR object detection results: A Vehicle is detected 21.48 meters away at a direction of 314.58 degrees. A Vehicle is detected 24.33 meters away at a direction of 336.26 degrees. A Vehicle is detected 11.60 meters away at a direction of 168.20 degrees. A Vehicle is detected 15.38 meters away at a direction of 188.91 degrees. A Vehicle is detected 23.36 meters away at a direction of 163.50 degrees.'
    prompt_sample24799_both = 'Given the following information captured from unit1, describe the physical environment this information provides and be as detailed as possible. Do your best to estimate distances of objects and their types, if there are any, and if there are any blockages between unit1 and unit2 from obstacles, cars, pedestrians or anything that can cause obstacles. Unit1 is a vehicle with a 360 camera (two 180 degree cameras), 360 degree 3D lidar, and a GPS module, unit2 is a vehicle mounted with a GPS module and a transmitter. The transmitter (Unit 2) is 10.539957539355683 meters away at a bearing of 159.9010115892885 degrees from the receiver (Unit 1). LiDAR object detection results: A Vehicle is detected 21.48 meters away at a direction of 314.58 degrees. A Vehicle is detected 24.33 meters away at a direction of 336.26 degrees. A Vehicle is detected 11.60 meters away at a direction of 168.20 degrees. A Vehicle is detected 15.38 meters away at a direction of 188.91 degrees. A Vehicle is detected 23.36 meters away at a direction of 163.50 degrees. Scene details from camera: The scene depicts a car driving underneath a bridge on a sunny day. There are several cars parked along the side of the road, and a traffic light can be seen in the distance. The camera is positioned from the front of the car, providing a bird''s-eye view of the surrounding area.In the scene, there are several cars driving on a busy city street underneath an overpass. The view is captured from the back of a car, providing a bird''s-eye view of the traffic below. There are multiple cars visible in the image, some of which are parked on the side of the road, while others are moving along the street. Traffic lights can be seen in the distance, indicating the flow of traffic and the need for drivers to follow the rules of the road. Overall, the scene captures the hustle and bustle of a busy city street, showcasing the diverse range of vehicles and pedestrians navigating their way through the urban environment.'

    # prompt = "Given the following information captured from unit1, describe all the details about the physical and network environment, such as observations on power and obstructed signals, this information provides and be as detailed as possible, and do your best to estimate distances of objects and their types, if there are any, and how any of the provided information affects the network if at all. Unit1 is a vehicle with a 360 camera, 360 degree 3D lidar, and 4 60GHz receiver phased arrays and a GPS module, unit2 is a vehicle mounted with a GPS module and a transmitter. The transmitter (Unit 2) is 8.08363553211514 meters away at a bearing of 182.445600684427 degrees from the receiver (Unit 1). LiDAR object detection results: A Vehicle is detected 11.37 meters away at a direction of 229.76 degrees. A Vehicle is detected 16.97 meters away at a direction of 181.48 degrees. A Vehicle is detected 9.83 meters away at a direction of 241.20 degrees. Average measured power from front receiver is 0.0001926337279201107, the right receiver is 0.00025486784215900116, the back receiver is 0.036197169767547166, and the left receiver is 0.00014873798556891416"
    # prompt = "Tell me how to configure the NR Function based on the Ericsson documentation and show me how to adjust it using MoShell, and if there are any examples, please reference and show me those examples"
    # response_mistral = chain_mistral.invoke(prompt_sensing)
    # print("Mistral Response:\n" + response_mistral['answer'])
    response_enwar = chain_llama.invoke(prompt)
    response_llama = llm_llama.invoke(prompt)
    # response_llama = llm_llama.invoke(prompt_sample24799_text_gps_only)
    print("Mistral Response:\n" + response_llama + '\n')
    print("Enwar Response:\n" + response_enwar['answer'])
    # print("Llama Response:\n" + response_llama)


if __name__ == "__main__":
    main()
