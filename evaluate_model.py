from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Load trained model
model = load_model("breast_cancer_model.h5")

# Load test images
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    r'C:\Users\ACER\Downloads\Dataset_BUSI_with_GT\test'
,                # <-- path to your test data
    target_size=(224, 224),        # same size as used in training
    batch_size=32,
color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

# Predict on test data
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# Confusion Matrix
print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
