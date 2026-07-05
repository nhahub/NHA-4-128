import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model, val_generator):
    val_generator.reset()
    class_names = list(val_generator.class_indices.keys())
    
    y_pred_probs = model.predict(val_generator, verbose=1)
    
    if y_pred_probs.shape[1] == 1:
        y_pred = (y_pred_probs > 0.5).astype("int32").flatten()
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)
        
    y_true = val_generator.classes
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={"size": 14})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix Heatmap\nAccuracy: {accuracy*100:.2f}%', fontsize=15)
    plt.show()

def predict_on_validation(model, val_generator, target_size=(150, 150)):
    correct_path = val_generator.directory
    test_datagen = ImageDataGenerator(rescale=1./255)
    temp_generator = test_datagen.flow_from_directory(
        correct_path,
        target_size=target_size,
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    true_labels = temp_generator.classes
    pred_probs = model.predict(temp_generator, verbose=1)
    pred_labels = (pred_probs > 0.5).astype("int32")
    return true_labels, pred_labels