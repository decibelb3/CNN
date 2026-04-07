import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import os



def create_cnn_model(input_shape, num_classes):
    """Create and compile the CNN model."""
    gpus = tf.config.list_physical_devices('GPU')

    print("TensorFlow version:", tf.__version__)
    print("GPUs detected:", len(gpus))
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu.name}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"   Name: {details.get('device_name', 'Unknown')}")
            print(f"   Compute capability: {details.get('compute_capability', 'N/A')}")
        except:
            pass

    if gpus:
        try:
            # Restrict to the first GPU (usually /GPU:0, which should be your 5070 Ti if only one card)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print("Restricted TensorFlow to GPU 0")
            # Optional: limit memory growth to avoid OOM on large models
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print("GPU restriction failed:", e)
    else:
        print("No GPUs available → falling back to CPU")
    
    
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax')) 

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train_categorical, X_val, y_val_categorical, epochs=10, batch_size=512):
    """Train the CNN model."""
    model.fit(X_train, y_train_categorical, epochs=epochs, batch_size=batch_size, 
              validation_data=(X_val, y_val_categorical))
    return model