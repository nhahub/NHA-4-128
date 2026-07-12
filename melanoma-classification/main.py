import os
from src.config import SAVE_DIR, EPOCHS_MODEL_2, EPOCHS_MODEL_3
from src.data_loader import get_data_generators
from src.models.vanilla_cnn import build_vanilla_cnn
from src.models.custom_cnn import build_custom_cnn
from src.models.mobilenet_model import build_mobilenet_model
from src.training import get_callbacks, train_model
from src.evaluation import evaluate_model, predict_on_validation
from src.utils import plot_history

def main():
    # Ensure save directory exists (for Kaggle environment)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Data Loading (Base)
    print("Loading base data generators...")
    train_generator, val_generator = get_data_generators(augment=False)

    # 2. Model 1: Vanilla CNN
    model_1 = build_vanilla_cnn()
    
    # 3. Model 2: Custom CNN
    print("Building and training Custom CNN (Model 2)...")
    train_generator_2, val_generator_2 = get_data_generators(augment=True)
    model_2 = build_custom_cnn()
    callbacks_list = get_callbacks('cnn_ex2')
    history_ex2 = train_model(model_2, train_generator_2, val_generator_2, EPOCHS_MODEL_2, callbacks_list)
    plot_history(history_ex2, "Model 2")

    # 4. Model 3: MobileNetV2
    print("Building and training MobileNetV2 (Model 3)...")
    model_3 = build_mobilenet_model(train_generator)
    early_stopping = get_callbacks('MobileNetV2')[1] # Extracting just early stopping as in notebook
    history_ex3 = train_model(model_3, train_generator, val_generator, EPOCHS_MODEL_3, [early_stopping])
    
    # Preserving exact Kaggle save path
    model_3.save(os.path.join(SAVE_DIR, 'MobileNetV2_model.h5'))
    print("Model saved successfully!")
    plot_history(history_ex3, "Model 3")

    # 5. Evaluation
    print("Evaluating Model 3...")
    evaluate_model(model_3, val_generator)

    # 6. Prediction on Validation Set (using Model 2)
    print("Predicting on validation set with Model 2...")
    true_labels, pred_labels = predict_on_validation(model_2, val_generator)

if __name__ == "__main__":
    main()