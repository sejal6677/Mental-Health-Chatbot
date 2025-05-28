try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    print("TensorFlow Keras import successful")
except Exception as e:
    print(f"Error importing TensorFlow Keras: {e}")
