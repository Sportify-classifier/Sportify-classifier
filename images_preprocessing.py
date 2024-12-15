import os
import matplotlib.pyplot as plt
from PIL import Image

# Chemin du dossier temporaire
temp_dir = "./data/all_data"

# Dictionnaire pour stocker les comptes par catégorie
category_counts = {}

# Contrôler les dimensions des images
image_dimensions = (224, 224)
incorrect_images = []

# Fonction pour corriger les images incorrectes
def correct_image(image_path):
    try:
        with Image.open(image_path) as img:
            # Convertir les images en mode RGB si elles sont en noir et blanc
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Redimensionner l'image
            img = img.resize(image_dimensions)
            # Sauvegarder l'image corrigée au format JPEG
            img.save(image_path, format='JPEG')
            print(f"Image corrigée : {image_path}")
    except Exception as e:
        print(f"Erreur lors de la correction de l'image {image_path}: {e}")

# Compter les images dans chaque catégorie
for category in os.listdir(temp_dir):
    category_path = os.path.join(temp_dir, category)
    if os.path.isdir(category_path):  # Vérifie que c'est un dossier
        num_files = 0
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            try:
                with Image.open(image_path) as img:
                    if img.format != 'JPEG' or img.size != image_dimensions or img.mode != 'RGB':
                        incorrect_images.append(image_path)
                    else:
                        num_files += 1
            except Exception as e:
                print(f"Erreur lors de l'ouverture de l'image {image_path}: {e}")
        category_counts[category] = num_files

# Corriger les images incorrectes
for image_path in incorrect_images:
    correct_image(image_path)

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
    
    if incorrect_images:
        # Images incorrectes
        f.write("\nImages corrigées :\n")
        for incorrect_image in incorrect_images:
            f.write(f"- {incorrect_image}\n")
    else:
        f.write("\nToutes les images sont au bon format et aux bonnes dimensions.\n")

print(f"Les résultats ont été sauvegardés dans le fichier : {output_file}")

# Générer un histogramme pour visualiser la répartition des catégories
categories = list(category_counts.keys())
counts = list(category_counts.values())

# Trier les catégories par nombre d'images
top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]  # Top 10
bottom_categories = sorted(category_counts.items(), key=lambda x: x[1])[:10]  # Bottom 10

# Combiner les catégories avec les moins nombreuses en haut et les plus nombreuses en bas
selected_categories = bottom_categories + top_categories[::-1]
selected_labels = [cat[0] for cat in selected_categories]
selected_counts = [cat[1] for cat in selected_categories]

# Définir les couleurs pour le top et le bottom
bottom_color = "#B266FF"  # Purple in hex
top_color = "#3399FF"    # Blue in hex
colors = [bottom_color] * len(bottom_categories) + [top_color] * len(top_categories)

# Générer un graphique avec des couleurs distinctes pour le top et le bottom
plt.figure(figsize=(10, 6))
plt.barh(selected_labels[::-1], selected_counts[::-1], color=colors[::-1])
plt.barh(selected_labels, selected_counts, color=colors)
plt.xlabel("Number of images")
plt.ylabel("Categories")
plt.title("Top and bottom 10 categories by number of images")

# Ajouter une légende avec des couleurs correspondantes
plt.legend(["Top 10 categories", "Bottom 10 categories"], loc='upper right')
plt.tight_layout()
plt.show()