
import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model

def train_model(
    model, epochs, batch_size, learning_rate, loss_func, version, train_images, train_col_images,
    train_features, train_labels, test_images, test_col_images, test_features, test_labels,
    load_best_weights=True,
    save_full_model=False
):

    """
    Trains a Keras model using provided training parameters.

    Parameters
    ----------
    model : tf.keras.Model
        The compiled model to train.
    epochs : int
        Number of training epochs.
    batch_size : int
        Size of the training batches.
    learning_rate : float
        Learning rate for the optimizer.
    loss_func : str or callable
        Loss function to use.
    version : str
        Version tag to name model checkpoints.
    train_images, train_col_images : np.ndarray
        Input image data arrays for training.
    train_features : np.ndarray
        Feature vector for training.
    train_labels : np.ndarray
        Target values for training.
    test_images, test_col_images : np.ndarray
        Input image data arrays for validation.
    test_features : np.ndarray
        Feature vector for validation.
    test_labels : np.ndarray
        Target values for validation.
    checkpoint_base_dir : str, optional
        Directory to save model checkpoints.
    load_best_weights : bool, optional
        Whether to load the best model weights after training.
    save_full_model : bool, optional
        Whether to save the entire model after training.

    Returns
    -------
    history : keras.callbacks.History
        Training history object.
    model : tf.keras.Model
        The trained (and possibly reloaded) model.
    """

    # Setup learning rate scheduler and early stopping
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.9,
        patience=30,
        verbose=1,
        mode="auto",
        cooldown=0,
        min_lr=1e-5
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0003,
        patience=100
    )

    # Create checkpoint directory
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PACKAGE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
    checkpoint_dir = os.path.join(PACKAGE_ROOT, "models", "from_train", "checkpoints", version)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model_weights.h5")


    # Save only the best weights
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    print('>> Training starting...')

    # Train model
    history = model.fit(
        x=[train_images, train_col_images, train_features],
        y=train_labels,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop, reduce_lr],
        validation_data=([test_images, test_col_images, test_features], test_labels)
    )

    print('>> Training finished.')

    # Optionally save the full model
    if save_full_model:
        full_model_path = os.path.join(checkpoint_dir, "full_model.h5")
        model.save(full_model_path)
        print(f">> Full model saved to {full_model_path}")

    # Optionally reload best weights
    if load_best_weights:
        print(">> Loading best model weights...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=loss_func,
            metrics=['accuracy']
        )
        model.load_weights(checkpoint_path)
    else:
        print(">> Not loading best model weights.")

    return history, model



