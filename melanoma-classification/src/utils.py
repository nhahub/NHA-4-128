import numpy as np
import matplotlib.pyplot as plt

def plot_history(history, title):
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    
    index_acc = np.argmax(val_acc)
    val_highest = val_acc[index_acc]
    
    Epochs = [i+1 for i in range(len(train_acc))]
    
    loss_label = f'Best epochs = {str(index_loss +1)}'
    acc_label = f'Best epochs = {str(index_acc + 1)}'
    
    plt.figure(figsize= (20,8))
    plt.style.use('fivethirtyeight')
    
    plt.subplot(1,2,1)
    plt.plot(Epochs , train_loss , 'r' , label = 'Training Loss')
    plt.plot(Epochs , val_loss , 'g' , label = 'Validation Loss')
    plt.scatter(index_loss + 1 , val_lowest , s = 150 , c = 'blue', label = loss_label)
    plt.title(f'Training and Validation Loss ({title})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(Epochs , train_acc , 'r' , label = 'Training Accuracy')
    plt.plot(Epochs , val_acc , 'g' , label = 'Validation Accuracy')
    plt.scatter(index_acc + 1 , val_highest , s = 150 , c = 'blue', label = acc_label)
    plt.title(f'Training and Validation Accuracy ({title})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()