import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data import Data, emotion_labels
from model import get_model

data = Data()

value_counts = data.train_data.label.value_counts()
colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

fig, ax = plt.subplots()
ax.bar(value_counts.index.values, value_counts.values, color=colors)
plt.show()

X_train = np.stack(data.train_data["pixels"].values).astype("float32")
y_train = np.stack(data.train_data["encoding"].values).astype("float32")

X_test = np.stack(data.test_data["pixels"].values).astype("float32")
y_test = np.stack(data.test_data["encoding"].values).astype("float32")


model = get_model()

model.fit(X_train, y_train, epochs=60, batch_size=32)

model.evaluate(X_test, y_test)

model.save("model_five.keras")

N = 4321

img_array = np.array([X_test[N]])

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

file_name = data.test_data["file_name"].values[N]
label = data.test_data["label"].values[N]

plt.imshow(Image.open(file_name), cmap="gray")
plt.title(label + " / " + emotion_labels[predicted_class[0]])
plt.show()
