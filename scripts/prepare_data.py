import os
import shutil
import glob

# Configuration
SOURCE_BASE = r"d:\Projects 2025\Final Year projects\Vertual brain\data\datasets"
TARGET_BASE = r"d:\Projects 2025\Final Year projects\Vertual brain\data\training_unified"

# Mapping: Target Folder -> List of source patterns
MAPPING = {
    "Alzheimer_Mild": [
        os.path.join(SOURCE_BASE, "archive (4)", "AD_MildDemented", "*"),
        os.path.join(SOURCE_BASE, "archive (2)", "Alzheimer_MRI_4_classes_dataset", "MildDemented", "*"),
    ],
    "Alzheimer_Moderate": [
        os.path.join(SOURCE_BASE, "archive (4)", "AD_ModerateDemented", "*"),
        os.path.join(SOURCE_BASE, "archive (2)", "Alzheimer_MRI_4_classes_dataset", "ModerateDemented", "*"),
    ],
    "Alzheimer_VeryMild": [
        os.path.join(SOURCE_BASE, "archive (4)", "AD_VeryMildDemented", "*"),
        os.path.join(SOURCE_BASE, "archive (2)", "Alzheimer_MRI_4_classes_dataset", "VeryMildDemented", "*"),
    ],
    "Glioma": [
        os.path.join(SOURCE_BASE, "archive (4)", "BT_glioma", "*"),
    ],
    "Meningioma": [
        os.path.join(SOURCE_BASE, "archive (4)", "BT_meningioma", "*"),
    ],
    "Pituitary": [
        os.path.join(SOURCE_BASE, "archive (4)", "BT_pituitary", "*"),
    ],
    "MS": [
        os.path.join(SOURCE_BASE, "archive (4)", "MS", "*"),
    ],
    "Parkinson": [
        os.path.join(SOURCE_BASE, "archive (3)", "parkinsons_dataset", "parkinson", "*"),
    ],
    "Normal": [
        os.path.join(SOURCE_BASE, "archive (4)", "Normal", "*"),
        os.path.join(SOURCE_BASE, "archive (3)", "parkinsons_dataset", "normal", "*"),
    ]
}

def consolidate():
    if os.path.exists(TARGET_BASE):
        print(f"Cleaning existing target directory: {TARGET_BASE}")
        shutil.rmtree(TARGET_BASE)
    
    os.makedirs(TARGET_BASE, exist_ok=True)

    for target_class, sources in MAPPING.items():
        class_target_dir = os.path.join(TARGET_BASE, target_class)
        os.makedirs(class_target_dir, exist_ok=True)
        
        count = 0
        for pattern in sources:
            files = glob.glob(pattern)
            for f in files:
                if os.path.isfile(f):
                    # Use a unique name to avoid collisions
                    filename = os.path.basename(f)
                    dest = os.path.join(class_target_dir, f"{count}_{filename}")
                    shutil.copy2(f, dest)
                    count += 1
        
        print(f"Class '{target_class}': Consolidated {count} images.")

if __name__ == "__main__":
    consolidate()
