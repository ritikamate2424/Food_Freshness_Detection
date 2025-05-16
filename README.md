# Food_Freshness_Detection

# 🥗 Food Freshness Detection Using CNN

# 📌 Overview

This project focuses on detecting the **freshness of fruits** using deep learning. It classifies fruits as **fresh** or **rotten** across different categories like apples, bananas, and oranges. The model is trained using a custom dataset and deployed with a clean, interactive **Streamlit UI** for end users to test images in real time.


# 🚀 Technologies Used

* Python 3.x
* TensorFlow / Keras
* NumPy & Pandas
* PIL for image handling
* Streamlit for deployment


# 🧠 CNN Model Summary

* **Model Type:** Sequential CNN
* **Input Shape:** 128x128 RGB Images
* **Classes:**

  * freshapples
  * freshbanana
  * freshoranges
  * rottenapples
  * rottenbanana
  * rottenoranges
* **Output Layer:** Softmax with 6 neurons
* **Loss Function:** Categorical Crossentropy
* **Optimizer:** Adam


# 📁 Dataset Details

* Dataset consists of 6 labeled folders for classification
* Preprocessed by:

  * Resizing to 128x128 pixels
  * Normalization (scaling pixel values between 0-1)
  * Augmentation if required (rotation, flip, etc.)

# 📊 Evaluation Metrics

```python
model.evaluate(test_generator)
```

* Accuracy: High classification accuracy on fresh vs rotten detection
* Visual validation using prediction confidence on test samples



# 💻 Streamlit UI Highlights

```python
st.title("🍎🍭 Fruit Freshness Detector")
st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])
...
prediction = model.predict(img_array)
class_names = [ ... ]
st.success(f"Prediction: {class_names[predicted_class]}")
```

* Upload any fruit image
* Real-time prediction output
* Confidence level displayed in percentage


# 📈 Sample Results

| Image         | Prediction   | Confidence |
| ------------- | ------------ | ---------- |
| Apple         | freshapples  | 98.23%     |
| Rotten Banana | rottenbanana | 94.51%     |



# 📎 Folder Structure

```
├── model/
│   └── fruit_freshness_model.keras
├── app/
│   └── app1.py
├── notebooks/
│   └── foodfreshness.ipynb
├── dataset/
│   ├── freshapples/
│   ├── freshbanana/
│   └── ...
└── README.md
```


# 💡 Future Enhancements

* Add more fruit classes (e.g., grapes, mango)
* Improve model with transfer learning (e.g., MobileNetV2)
* Integrate voice-based predictions
* Deploy with Docker or AWS Lambda
* Add heatmap visualizations (e.g., Grad-CAM)

