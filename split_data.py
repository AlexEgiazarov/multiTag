import argparse
import csv
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

#saves the csv file with given data
def save_csv(data, path, fieldnames=['image_path', 'gender', 'articleType', 'baseColour']):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(dict(zip(fieldnames, row)))

if __name__ == '__main__':

    #arguments
    parser = argparse.ArgumentParser(description='Split data for the dataset')
    parser.add_argument('--input', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--output', type=str, required=True, help="Path to the working folder")

    args = parser.parse_args()
    input_folder = args.input
    output_folder = args.output
    annotation = os.path.join(input_folder, 'styles.csv')

    #opening annotation file

    all_data= []

    #open annotation file
    with open(annotation) as csv_file:
        #create reader
        reader = csv.DictReader(csv_file)
        #parser csv
        for row in tqdm(reader, total=reader.line_num):
            #we need image id to build the path ot the image file
            img_id = row['id']
            #using only 3 attributes
            gender = row['gender']
            articleType = row['articleType']
            baseColour = row['baseColour']
            img_name = os.path.join(input_folder, 'images', str(img_id)+'.jpg')
            #check if file is in place
            if os.path.exists(img_name):
                #check if the image has 80*60 pixels with 3 channels
                img= Image.open(img_name)
                if img.size == (60, 80) and img.mode == 'RGB':
                    all_data.append([img_name, gender, articleType, baseColour])

    #set the seeed of the random numbers generator, so we can reproduce the results later
    np.random.seed(42)
    #constuct a numpy array from the list
    all_data = np.asarray(all_data)
    #take 40000 samples in random order
    inds = np.random.choice(40000, 40000,replace=False)
    #split the data into train/val and save (probably can be done in pytorch)
    save_csv(all_data[inds][:32000], os.path.join(output_folder, 'train.csv'))
    save_csv(all_data[32000:40000], os.path.join(output_folder, 'val.csv'))