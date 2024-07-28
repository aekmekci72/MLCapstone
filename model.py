import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

df = pd.read_csv('/content/DiseaseAndSymptoms.csv')
df.head()

symptoms_cols = [col for col in df.columns if 'Symptom_' in col]
df['Symptoms'] = df[symptoms_cols].apply(lambda x: ','.join(x.dropna()), axis=1)

mlb = MultiLabelBinarizer()
df['Symptoms'] = df['Symptoms'].str.lower().str.strip()
symptoms_binarized = mlb.fit_transform(df['Symptoms'].str.split(','))

labels = pd.get_dummies(df['Disease'])

X_train, X_test, y_train, y_test = train_test_split(symptoms_binarized, labels, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(labels.shape[1], activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.2, callbacks=[early_stopping])
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")