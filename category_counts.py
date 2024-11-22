import os

# Chemin du dossier temporaire
temp_dir = "./data/all_data"

# Dictionnaire pour stocker les comptes par catégorie
category_counts = {}

# Compter les images dans chaque catégorie
for category in os.listdir(temp_dir):
    category_path = os.path.join(temp_dir, category)
    if os.path.isdir(category_path):  # Vérifie que c'est un dossier
        num_files = len(os.listdir(category_path))
        category_counts[category] = num_files

# Trouver le minimum et le maximum
min_count = min(category_counts.values())
max_count = max(category_counts.values())
min_categories = [cat for cat, count in category_counts.items() if count == min_count]
max_categories = [cat for cat, count in category_counts.items() if count == max_count]

# Chemin du fichier de sortie
output_file = "./category_counts.txt"

# Écrire les résultats dans un fichier
with open(output_file, "w") as f:

    f.write(f"\nCatégorie avec le minimum d'images ({min_count} images) : {', '.join(min_categories)}\n")
    f.write(f"Catégorie avec le maximum d'images ({max_count} images) : {', '.join(max_categories)}\n")
    f.write("\n")
    
    # Résultats triés en ordre croissant
    f.write("Nombre d'images par catégorie (ordre croissant) :\n")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1]):
        f.write(f"- {category} : {count} images\n")

print(f"Les résultats ont été sauvegardés dans le fichier : {output_file}")