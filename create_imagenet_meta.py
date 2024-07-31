import os
import json
import pandas as pd

# Path to your dataset
dataset_path = "data/imagenet/images"

# Load JSON file containing class name to category ID mapping
with open('data/imagenet/classnames.json', 'r') as f:
    class_to_category = json.load(f)

# Initialize empty lists to store data
class_names = []
split_types = []
image_locations = []

# Iterate through train and validation directories
for split_type in ['train', 'val','test']:
# for split_type in ['val']:
    split_dir = os.path.join(dataset_path, split_type)
    class_folders = os.listdir(split_dir)
    
    # Iterate through class folders
    for class_folder in class_folders:
        class_dir = os.path.join(split_dir, class_folder)
        image_files = os.listdir(class_dir)
        
        # Append data to lists
        class_names.extend([class_folder] * len(image_files))
        split_types.extend([split_type] * len(image_files))
        image_locations.extend([os.path.join(class_dir, image_file) for image_file in image_files])

# Create DataFrame
df = pd.DataFrame({
    'label': class_names,
    'img_set': split_types,
    'img_path': image_locations
})

# Function to get category ID based on class name
def get_category_id(class_name):
    return class_to_category.get(class_name, None)

# Add category_id column
df['category_id'] = df['label'].apply(get_category_id)

df['img_path'] = df['img_path'].str.replace('data/imagenet/', '')

# Save DataFrame to CSV
df.to_csv('imagenet_dataset_info.csv', index=True)

print("DataFrame saved as CSV successfully.")
