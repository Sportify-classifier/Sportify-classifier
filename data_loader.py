import sys
from utils import prepare_data

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_loader.py <input_data_dir> <output_data_dir>")
        sys.exit(1)

    input_data_dir = sys.argv[1]
    output_data_dir = sys.argv[2]

    # Appeler la fonction pour préparer les données
    prepare_data(input_data_dir, output_data_dir)