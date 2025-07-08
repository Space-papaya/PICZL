import tensorflow as tf

def set_computing(memory_gb=10):
    """
    Detects available GPUs and configures TensorFlow to use the first GPU (if present),
    optionally limiting its memory usage to `memory_gb`.

    Args:
        memory_gb (int): Maximum amount of GPU memory to allocate (in GB).

    Returns:
        str: The name of the device used for computation (e.g., '/GPU:0' or 'CPU').
    """
    print("Available GPUs:", tf.config.list_physical_devices("GPU"))
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=1024 * memory_gb
                        )
                    ],
                )
            tf.config.set_visible_devices(gpus[0], "GPU")
            print("GPU enabled:", gpus[0])
        except RuntimeError as e:
            print("Failed to set GPU configuration:", e)
    else:
        print("No GPU available. Running on CPU.")

    device = "/GPU:0" if gpus else "CPU"
    print(f"Training on device: {device}")

    return device
