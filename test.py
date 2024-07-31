import pandas as pd
import json

# # Load the JSON file
# with open('/fsx/homes/Mohamed.Imam@mbzuai.ac.ae/projects/TAAL/data/oxford_flowers/cat_to_name.json', 'r') as file:
#     original_dict = json.load(file)
# swapped_dict = {value: key for key, value in original_dict.items()}

# df = pd.read_csv('/fsx/homes/Mohamed.Imam@mbzuai.ac.ae/projects/TAAL/data/oxford_flowers/oxford_flowers_meta.csv')
# # Update the label column
# df['category_id'] = df['label'].map(swapped_dict).astype(float)
# df = df[['img_id','img_name','img_path','category_id','label','img_set']]
# df.to_csv('oxford_flowers_meta.csv')

df = pd.read_csv('/fsx/homes/Mohamed.Imam@mbzuai.ac.ae/projects/TAAL/data/oxford_flowers/oxford_flowers_meta.csv')
print(df.shape)

import scipy.io

# Load the .mat file
mat_data = scipy.io.loadmat('/fsx/homes/Mohamed.Imam@mbzuai.ac.ae/projects/TAAL/data/oxford_flowers/imagelabels.mat')
df['category_id'] = mat_data['labels'].flatten()
# Display the keys (variables) in the .mat file
print(mat_data.keys())
breakpoint()
# To display a specific variable, access it using the variable name
print(mat_data['labels'].shape)
df = df[['img_id','img_name','img_path','category_id','label','img_set']]
df.to_csv('oxford_flowers_meta.csv')


# # Update the category_id column by add 1
# df['category_id'] = df['category_id'] + 1
# df = df[['img_id','img_name','img_path','category_id','label','img_set']]
# df.to_csv('oxford_flowers_meta.csv')

# Load the JSON data into a dictionary
# # Path to the JSON file
# json_file_path = '/fsx/homes/Mohamed.Imam@mbzuai.ac.ae/projects/TAAL/data/oxford_flowers/split_zhou_OxfordFlowers.json'

# # Load the JSON data from the file
# with open(json_file_path, 'r') as file:
#     data = json.load(file)
    
# # Function to increment integer values in the data
# def increment_integer_values(data):
#     for key, samples in data.items():
#         for sample in samples:
#             # Assuming the second element in each sample list is the integer to be incremented
#             if isinstance(sample[1], int):
#                 sample[1] += 1

# # Increment the integer values
# increment_integer_values(data)

# # Save the updated data
# # Save the updated data to a JSON file
# with open('oxford_flowers_split.json', 'w') as json_file:
#     json.dump(data, json_file, indent=2)