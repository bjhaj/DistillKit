import os

# Base directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model paths
TEACHER_MODEL_PATH = os.path.join(MODELS_DIR, 'teacher_model.pth')
STUDENT_MODEL_PATH = os.path.join(MODELS_DIR, 'student_model.pth')
SMALL_MODEL_PATH = os.path.join(MODELS_DIR, 'small_model.pth')
QUANTIZED_MODEL_PATH = os.path.join(MODELS_DIR, 'quantized.pth')

# Data paths
SOFT_TARGETS_PATH = os.path.join(DATA_DIR, 'cifar10_soft_targets.pt')
HARD_LABELS_PATH = os.path.join(DATA_DIR, 'cifar10_labels.pt')
TRAIN_IMAGES_PATH = os.path.join(DATA_DIR, 'cifar10_train_images.pt')

# Output paths
TRAINING_HISTORY_PATH = os.path.join(OUTPUT_DIR, 'training_history.json')
EVALUATION_RESULTS_PATH = os.path.join(OUTPUT_DIR, 'evaluation_results.json')

def get_model_path(model_type):
    """
    Get path for a specific model type.
    
    Args:
        model_type (str): 'teacher', 'student', or 'small'
        quantized (bool): Whether to get quantized model path
        
    Returns:
        str: Path to model file
    """
    if model_type == 'teacher':
        return TEACHER_MODEL_PATH
    elif model_type == 'student':
        return STUDENT_MODEL_PATH
    elif 'quant' in model_type:
        return QUANTIZED_MODEL_PATH
    elif model_type == 'small':
        return SMALL_MODEL_PATH
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_data_path(data_type):
    """
    Get path for a specific data type.
    
    Args:
        data_type (str): 'soft_targets', 'hard_labels', or 'train_images'
        
    Returns:
        str: Path to data file
    """
    paths = {
        'soft_targets': SOFT_TARGETS_PATH,
        'hard_labels': HARD_LABELS_PATH,
        'train_images': TRAIN_IMAGES_PATH
    }
    
    if data_type not in paths:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return paths[data_type] 