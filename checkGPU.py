#%%
# Print GPU Number
from tensorflow.python.client import device_lib
import tensorflow as tf

def getGPUName():
    print(device_lib.list_local_devices())
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
