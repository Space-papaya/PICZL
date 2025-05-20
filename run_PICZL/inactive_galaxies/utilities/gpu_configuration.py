
#importing libraries
import tensorflow as tf

# Use tf.config.experimental.set_virtual_device_configuration to allocate a specific amount of GPU memory to script
gpus = tf.config.experimental.list_physical_devices('GPU')
memory = 25  # Specify memory in GB

if gpus:
    for gpu in gpus:
        # Allocate a specific amount of memory (e.g., 30GB) to each GPU
        tf.config.experimental.set_virtual_device_configuration(
            gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * memory)]
        )
