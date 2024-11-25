import os
import shutil

# Chemins des dossiers
base_dir = "./data"
folders = ["train", "test", "valid"]
temp_dir = os.path.join(base_dir, "all_data")

# Étape 1 : Fusionner les données dans un seul dossier temporaire
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
            
            # Renommage des fichiers en fonction du dossier d'origine
            if folder == "train":
                # Format : 001.jpg, 002.jpg, etc.
                new_file_name = file.zfill(7)  # Ajoute des zéros pour avoir 001.jpg
            elif folder == "test":
                # Format : inchangé (1.jpg, 2.jpg, etc.).
                new_file_name = file
            elif folder == "valid":
                # Format : 01.jpg, 02.jpg, etc.
                base_name, ext = os.path.splitext(file)
                new_file_name = base_name.zfill(2) + ext  # Préfixe avec un zéro pour 1 -> 01.jpg
            
            dst = os.path.join(temp_category_path, new_file_name)
            shutil.copy2(src, dst)
