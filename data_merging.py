import os
import shutil

#PLUS BESOIN D'UTILISER CE FICHIER APRES AVOIR CREER LA PIPELINE

base_dir = "./data"
folders = ["train", "test", "valid"]
temp_dir = os.path.join(base_dir, "all_data")

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)
        temp_category_path = os.path.join(temp_dir, category)
        
        if not os.path.exists(temp_category_path):
            os.makedirs(temp_category_path)
        
        for file in os.listdir(category_path):
            src = os.path.join(category_path, file)
            
            if folder == "train":
                new_file_name = file.zfill(7)
            elif folder == "test":
                new_file_name = file
            elif folder == "valid":
                base_name, ext = os.path.splitext(file)
                new_file_name = base_name.zfill(2) + ext
            
            dst = os.path.join(temp_category_path, new_file_name)
            shutil.copy2(src, dst)
