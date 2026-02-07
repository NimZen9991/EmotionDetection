# EmotionDetection
# Facial Emotion Recognition Analysis 🎭

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-green)

A comparative study of Traditional Machine Learning versus Deep Learning (CNN) pipelines for classifying facial expressions into 8 distinct emotional categories. This project explores feature extraction, dimensionality reduction, embedding visualizations, and class imbalance handling.

## 📌 Project Overview

The objective of this project is to build a robust pipeline for facial emotion classification. The study contrasts the performance of standard classifiers (Logistic Regression, Random Forest, MLP) against a custom Convolutional Neural Network (CNN).

Key comparisons include:
* **Feature Space:** Raw pixels vs. PCA-reduced features.
* **Modeling:** Scikit-Learn classifiers vs. TensorFlow/Keras CNN.
* **Analysis:** Supervised metrics vs. Unsupervised clustering (t-SNE/UMAP) of learned embeddings.

## 📂 Dataset

**Source:** [AffectNet Training Data (Kaggle)](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data)

* **Total Images:** ~29,000 samples.
* **Resolution:** Resized to 96x96 grayscale.
* **Classes (8):** Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise.
* **Preprocessing:**
    * Images filtered using **relFCs** (Percentage of First Component) to remove low-detail/monochromatic images.
    * Converted to `.npz` format for efficient memory loading.

## ⚙️ Methodology

### 1. Data Engineering
* **Optimization:** Due to the large dataset size, raw images were converted to NumPy arrays and stored in compressed `.npz` format to reduce loading time from hours to seconds.
* **Normalization:** Pixel values scaled using `StandardScaler`.
* **Dimensionality Reduction:** Applied **PCA (1000 components)** for traditional ML models to reduce feature space while retaining variance.

### 2. Traditional Machine Learning
Baseline models were trained on PCA-reduced features:
* **Logistic Regression** (L2 penalty, balanced weights)
* **Random Forest** (300 estimators, max depth 20)
* **MLP Classifier** (3-layer Neural Network)

### 3. Deep Learning (CNN)
A custom CNN architecture was designed to handle high-dimensional spatial data:
* **Architecture:** 4 Convolutional Blocks (Conv2D -> ReLU -> MaxPool -> BatchNorm -> Dropout).
* **Regularization:** L2 Regularization, Dropout (0.2 - 0.6), and Early Stopping.
* **Augmentation:** Random rotation, zoom, shifting, and flipping to improve generalization.
* **Class Imbalance:** Handled using `compute_class_weight` to penalize misclassification of minority classes (e.g., Disgust).

### 4. Embedding Analysis
To understand *how* the model learns, high-dimensional embeddings were extracted from the MLP and visualized using:
* **K-Means Clustering**
* **t-SNE & UMAP** (Non-linear dimensionality reduction)

## 📊 Results & Performance

| Model | Accuracy (Validation) | Key Observation |
| :--- | :--- | :--- |
| **Random Forest** | ~39.2% | Struggled with high-dimensional feature complexity. |
| **Logistic Regression** | ~47.1% | Better baseline, but limited by linearity. |
| **MLP (Neural Net)** | ~48.2% | Best of the traditional models. |
| **Custom CNN** | **~56.5%** | **Best Performance.** Effectively captured spatial hierarchies. |

### Classification Insights (CNN)
* **High Performance:** `Happy` (F1: 0.86) and `Neutral` (F1: 0.81).
* **Challenges:** `Anger` and `Disgust` showed high confusion, likely due to subtle visual similarities and lower sample counts.

## 📈 Visualizations

### Confusion Matrix (CNN)
![Confusion Matrix](confusion_matrix.png)
> The model distinguishes positive emotions well but struggles to differentiate between negative emotions like Fear and Surprise.

### Training Dynamics
![Training Curves](accuracy_loss.png)
> The CNN shows steady learning with Early Stopping preventing significant overfitting.

### Embedding Space (t-SNE/UMAP)
![t-SNE Plot](tsne_clusters.png)
> Unsupervised visualization reveals distinct clusters for 'Happy' and 'Neutral', while negative emotions overlap significantly in the embedding space.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/emotion-recognition.git](https://github.com/yourusername/emotion-recognition.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy pandas tensorflow scikit-learn seaborn matplotlib
    ```
3.  **Load Data:**
    Ensure `gray_image_data.npz` is in the root directory (or update the path in the notebook).
4.  **Run the Notebook:**
    Open `Emotion_Recognition_Pipeline.ipynb` in Jupyter or Google Colab.

## 🧠 Future Work
* **Transfer Learning:** Implement ResNet50 or VGG16 to leverage pre-trained weights.
* **Fine-Grained Augmentation:** Specifically target under-represented classes (Anger, Disgust) with synthetic generation.
* **Ensemble Methods:** Combine CNN and SVM predictions for the confusion-heavy classes.

## 🤝 Credits
* **Author:** Nima
* **Data Source:** Kaggle / AffectNet
