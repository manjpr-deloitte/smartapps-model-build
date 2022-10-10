# -*- coding: utf-8 -*-


# import statements
import logging
import boto3
import argparse
import logging
import os
import pathlib
import pandas as pd
import subprocess
import sys
import configparser
import json

# Setting up logging functions
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

BUCKET_NAME = "refinedcsv"

def normalize_data(df,col_name,min_value,max_value):
    df[col_name]= df[col_name]-min_value
    df[col_name]= df[col_name]/(max_value-min_value)
    return df
                                                                             
if __name__ == "__main__":

    # subprocess.check_call([sys.executable, "-m", "pip", "install", "fsspec", "s3fs"])
    # subprocess.run([sys.executable, "-m", "pip", "install", "fsspec"])
    subprocess.run([sys.executable, "-m", "pip", "install", "s3fs==2022.2.0"])
    
    
    base_dir = "/opt/ml/processing" # Stored on local
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    
    processed_file_name = "processed_file.csv"
    ##New data changes
    columns_to_consider = ["popularity",'Action',"Adventure","Animation","Comedy","Crime",
                           "Documentary","Drama","Family","Fantasy","Foreign","History","Horror","Music","Mystery","Romance","science_fiction",
                           "TV_Movie","Thriller","War","Western"]
    
    print("Starting preprocessing")
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--output-data", type=str, required=True)
    args = parser.parse_args()
    
    # Uncomment for deafult
    # pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    
    input_data = args.input_data
    output_data = args.output_data
    bucket = input_data.split("/")[2]
    prefix = input_data.split("/")[3]
    output_filepath = output_data.split("/")[-2] + "/" + "processed.csv"
    # s3_client = boto3.client('s3')
    # response = s3_client.list_objects_v2(Bucket=bucket)
    # objects = sorted(response['Contents'], key=lambda obj: obj['LastModified'])
    # part_objects = []
    # for i in objects:
    #     if "part" in i['Key']:
    #         part_objects.append(i)
    # # latest_object = part_objects[-1]['Key']
    # filename = "csv/part-00000-d409a910-40d0-4943-991b-ce23c113dec8-c000.csv"
    # logger.info("latest_object: %s", latest_object)
    # filename = prefix +'/'+ latest_object[latest_object.rfind('/')+1:] # Remove path
    filename = "csv/part-00000-d409a910-40d0-4943-991b-ce23c113dec8-c000.csv"
#     filename = "dataset/movie_data.csv"
    logger.info("filename: %s",filename)
    logger.info("Processing file %s from bucket: %s",filename ,  bucket)
    fn = f"{base_dir}/data/abalone-dataset.csv"
    my_east_session = boto3.Session(region_name = 'us-east-1')
    s3 = my_east_session .resource("s3")
    s3.Bucket(bucket).download_file(filename, fn)
    result = s3.Bucket('refinedcsv').upload_file(fn, '/dataset/file.csv')

    print(result)
    

    
    data = pd.read_csv(fn)
    # data.sort_values("MovieId",inplace = True)
    # data = data.set_index("MovieId")
    data = data[columns_to_consider]
    genre_column = [e for e in columns_to_consider if e not in ('popularity',"title")]
    print(f'genre: {genre_column}')
    #data[genre_column] = data[genre_column].replace([1], [999])
    
    ##New data changes
    # mean_year = data["Movie_Year"].mean()
    # data["Movie_Year"] = data["Movie_Year"].fillna(mean_year)
    
    
#     fn_min_max = f"{base_dir}/data/min_max_algo_variables.json"
#     s3.Bucket(bucket).download_file("dataset/min_max_algo_variables.json", fn_min_max)
#     fn_min_max = open(fn_min_max, "r")
#     fn_min_max = json.loads(fn_min_max.read())
#     print(fn_min_max)
#     # data = normalize_data(data,"Movie_Year",fn_min_max["min-year"],fn_min_max["max-year"])
#     data = normalize_data(data,"popularity",fn_min_max["min-popularity"],fn_min_max["max-popularity"])
    

#     print(data.index.name)
    data.to_csv(fn,index = False)
    s3 = boto3.resource("s3")
#     #s3.Bucket("smartapps-studio-model-building-bucket").upload_file(fn, "data/processed.csv")
    s3.Bucket(BUCKET_NAME).upload_file(fn, output_filepath)
    s3.upload_file(fn, "refinedcsv", "data/processed.csv")
    print("File preprocessing complete")
    # print(data.info())
    # print(data.count())
    
    
    