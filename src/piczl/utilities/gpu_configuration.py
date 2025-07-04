
import tensorflow as tf

def limit_gpu_memory(memory_gb=25):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * memory_gb)]
                )
        except RuntimeError as e:
            print(f"Error setting GPU memory limit: {e}")
