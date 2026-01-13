import os
import shutil
from pathlib import Path

# Définir les classes dans l'ordre correct
CLASS_MAPPING = {
    0: "100fc",
    1: "50fc",
    2: "200fc",
    3: "500fc",
    4: "1000fc",
    5: "5000fc",
    6: "10000fc",
    7: "20000fc",
    8: "100$",
    9: "5$",
    10: "10$",
    11: "50$",
    12: "20$",
    13: "1$"
}

# Mapping des variantes de noms de classes trouvées dans les datasets
CLASS_VARIANTS = {
    "100fc": ["100fc", "100_Fc", "100FC", "100 franc"],
    "50fc": ["50fc", "50_Fc", "50FC", "50 franc"],
    "200fc": ["200fc", "200_Fc", "200FC", "200 franc"],
    "500fc": ["500fc", "500_Fc", "500FC", "500 franc"],
    "1000fc": ["1000fc", "1000_Fc", "1000FC", "1000 franc"],
    "5000fc": ["5000fc", "5000_Fc", "5000FC", "5000 franc"],
    "10000fc": ["10000fc", "10000_Fc", "10000FC", "10000 franc"],
    "20000fc": ["20000fc", "20000_Fc", "20000FC", "20000 franc"],
    "100$": ["100$", "100_$", "100 $", "100dollars"],
    "5$": ["5$", "5_$", "5 $", "5dollars"],
    "10$": ["10$", "10_$", "10 $", "10dollars"],
    "50$": ["50$", "50_$", "50 $", "50dollars"],
    "20$": ["20$", "20_$", "20 $", "20dollars"],
    "1$": ["1$", "1_$", "1 $", "1dollars"]
}

# Créer le mapping inverse pour trouver la classe correcte
reverse_mapping = {}
for canonical, variants in CLASS_VARIANTS.items():
    for variant in variants:
        reverse_mapping[variant.lower()] = canonical

# Datasets sources
SOURCE_DATASETS = [
    r"D:\DATASET\BILLET G5",
    r"D:\DATASET\DataSetAll",
    r"D:\DATASET\DatasetImage",
    r"D:\DATASET\DATASET_BILLETS_ARGENT",
    r"D:\DATASET\dataset_money groupe 5"
]

# Dossier de destination
OUTPUT_DIR = r"D:\DATASET\DATASET_CONSOLIDÉ"

def find_correct_class(folder_name):
    """Trouver la classe correcte pour un dossier donné"""
    normalized = folder_name.lower().strip()
    
    # Vérifier les variantes
    if normalized in reverse_mapping:
        return reverse_mapping[normalized]
    
    return None

def consolidate_datasets():
    """Consolider tous les datasets dans un nouveau dossier"""
    
    # Créer le dossier de destination
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Créer les dossiers de classes
    for class_name in CLASS_MAPPING.values():
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    total_files = 0
    files_by_class = {class_name: 0 for class_name in CLASS_MAPPING.values()}
    
    # Traiter chaque dataset source
    for source_dataset in SOURCE_DATASETS:
        if not os.path.exists(source_dataset):
            print(f"⚠️  Dataset non trouvé: {source_dataset}")
            continue
        
        print(f"\n📂 Traitement de: {os.path.basename(source_dataset)}")
        
        # Lister les dossiers de classes
        for class_folder in os.listdir(source_dataset):
            source_class_path = os.path.join(source_dataset, class_folder)
            
            if not os.path.isdir(source_class_path):
                continue
            
            # Trouver la classe correcte
            correct_class = find_correct_class(class_folder)
            
            if correct_class is None:
                print(f"  ⚠️  Classe non reconnue: {class_folder}")
                continue
            
            # Dossier de destination pour cette classe
            dest_class_path = os.path.join(OUTPUT_DIR, correct_class)
            
            # Copier les images
            image_count = 0
            for image_file in os.listdir(source_class_path):
                source_file = os.path.join(source_class_path, image_file)
                
                if os.path.isfile(source_file):
                    # Créer un nom unique pour éviter les doublons
                    dest_file = os.path.join(dest_class_path, image_file)
                    
                    # Si le fichier existe déjà, ajouter un suffixe
                    if os.path.exists(dest_file):
                        base, ext = os.path.splitext(image_file)
                        counter = 1
                        while os.path.exists(dest_file):
                            dest_file = os.path.join(dest_class_path, f"{base}_{counter}{ext}")
                            counter += 1
                    
                    try:
                        shutil.copy2(source_file, dest_file)
                        image_count += 1
                    except Exception as e:
                        print(f"    ❌ Erreur lors de la copie: {source_file} -> {e}")
            
            if image_count > 0:
                print(f"  ✅ {class_folder} → {correct_class}: {image_count} images copiées")
                files_by_class[correct_class] += image_count
                total_files += image_count

    # Afficher le résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DE LA CONSOLIDATION")
    print("="*60)
    print(f"\n📁 Dossier de destination: {OUTPUT_DIR}\n")
    
    for idx, (class_idx, class_name) in enumerate(CLASS_MAPPING.items()):
        count = files_by_class[class_name]
        print(f"{idx}: {class_name:15} - {count:5} images")
    
    print("\n" + "="*60)
    print(f"✅ Total: {total_files} images consolidées")
    print("="*60)

if __name__ == "__main__":
    print("🚀 Démarrage de la consolidation des datasets...\n")
    consolidate_datasets()
    print("\n✨ Consolidation terminée!")
