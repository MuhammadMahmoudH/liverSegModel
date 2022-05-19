import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

# summarize history for acc
def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# summarize history for loss
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# plotting model structure
def pltModel(model,to_file ,show_shapes):
    plot_model(model, to_file=to_file, show_shapes=show_shapes, show_layer_names=True)
    model_name = type(model).__name__
    print(f'----------- Model {to_file} Plotted -------------')
