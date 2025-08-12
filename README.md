# ♻ Waste Material Segregation using CNN

## 📌 Overview
This project implements a **Convolutional Neural Network (CNN)** to automatically classify waste materials into 7 categories using image data.  
The goal is to aid waste management and recycling by providing an accurate automated waste segregation model.

**Classes:**
- Cardboard  
- Food Waste  
- Glass  
- Metal  
- Other  
- Paper  
- Plastic  

---

## 🎯 Objectives
- Accurately classify waste materials into predefined categories.
- Improve waste segregation efficiency for better recycling.
- Demonstrate CNN-based image classification techniques with augmentation and normalization.
- Handle dataset imbalance using class weights.

---

## 📂 Dataset
- Dataset consists of **images grouped into folders** representing their respective waste categories.
- Each folder contains raw images in different dimensions.
- Images are preprocessed — **resized to 96×96 pixels** and normalized to [0,1] range.

---

## ⚙️ Tech Stack
- **Python 3.10+**
- **TensorFlow / Keras** for deep learning
- **NumPy, Pandas** for data handling
- **Matplotlib, Seaborn** for visualization
- **scikit-learn** for preprocessing & model evaluation
- **PIL** for image manipulation

---

## 🛠 Project Workflow
### 1️⃣ Data Loading & Preprocessing
- Load images from folder structure.
- Resize images to 96×96, normalize pixel values.
- Encode string labels to numerical values.
- Split dataset (80% train, 20% test/validation).

### 2️⃣ Data Visualization
- Bar chart showing **class distribution**.
- Display **sample images** for each class.

### 3️⃣ Model Architecture
CNN with:
- **3 convolutional blocks** (Conv2D + Batch Normalization + MaxPooling + Dropout)
- Global Average Pooling
- Fully connected Dense layers
- Softmax output for multi-class classification

### 4️⃣ Training & Optimization
- Optimizer: **Adam**
- Loss: **Categorical Crossentropy**
- Metrics: Accuracy
- Callbacks: ReduceLROnPlateau, EarlyStopping
- Handled class imbalance with **class weights**.

### 5️⃣ Data Augmentation
- Applied **rotation, zoom, shifts, flips** using `ImageDataGenerator`.
- Retrained model with augmented data to improve generalization.

### 6️⃣ Evaluation
- Measured validation accuracy and loss.
- Generated predictions and evaluated performance.

---

## 📊 Results
| Model Training | Validation Accuracy | Validation Loss |
|----------------|--------------------|-----------------|
| Base Model     | 49.5%              | 1.4196          |
| With Augmentation | **55.1%**        | 1.5096          |

**Key Insights:**
- Data augmentation reduced overfitting and improved validation accuracy.
- Normalization & Batch Normalization stabilized training.
- Model performs reasonably well but could benefit from more data or advanced architectures (e.g., Transfer Learning).

---

5. **Train & evaluate model** inside the notebook.

---

## 📈 Possible Improvements
- Use **Transfer Learning** (e.g., ResNet, EfficientNet) for higher accuracy.
- Incorporate **fine-tuning** and **learning rate scheduling**.
- Experiment with **more augmentation techniques**.
- Deploy as a **web app** for real-time classification.
