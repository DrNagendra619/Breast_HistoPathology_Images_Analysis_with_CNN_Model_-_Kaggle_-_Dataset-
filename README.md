# Breast_HistoPathology_Images_Analysis_with_CNN_Model_-_Kaggle_-_Dataset-
Breast_HistoPathology_Images_Analysis_with_CNN_Model_[_Kaggle_=_Dataset]
# 🔬 Breast Histopathology Image Analysis with CNN

This repository contains a **Jupyter Notebook** pipeline that demonstrates how to build, train, and evaluate a **Convolutional Neural Network (CNN)** for the binary classification of **breast cancer histopathology images**. The model is trained to differentiate between image patches containing **invasive ductal carcinoma (IDC) (Malignant)** and those without (**Benign/Normal)**.

The pipeline utilizes the popular **PatchCamelyon (PCam) / Breast Histopathology Images Kaggle Dataset**.

## 🚀 Key Features

* **Deep Learning Classification:** Implements a custom **Convolutional Neural Network (CNN)** using Keras/TensorFlow.
* **Image Data Handling:** Utilizes `ImageDataGenerator` for efficient loading, normalization, and **Data Augmentation** (rescaling, zooming, flipping) to prevent overfitting.
* **Kaggle Dataset Integration:** Designed to work with the standard structure of the PatchCamelyon/Breast Histopathology Images dataset (typically two folders containing image patches).
* **Model Training:** Includes training steps with appropriate callbacks (e.g., EarlyStopping) to optimize performance.
* **Performance Visualization:** Generates and plots essential metrics:
    * **Loss and Accuracy Curves** (Training vs. Validation).
    * **Confusion Matrix** (Heatmap).
    * **ROC Curve** and **AUC** score.
* **Classification Report:** Provides a final, detailed summary of Precision, Recall, and F1-Score.

---

## 🔬 Analysis Overview

| Component | Method / Tool | Purpose |
| :--- | :--- | :--- |
| **Dataset** | Breast Histopathology Images (Kaggle/PCam) | Image patches of lymph node sections for IDC detection. |
| **Model** | Convolutional Neural Network (CNN) | Feature extraction and classification optimized for image data. |
| **Preprocessing** | Data Augmentation | Increases the diversity of the training set to improve generalization. |
| **Classification** | Binary Classification (Malignant/Benign) | Predicts the presence (1) or absence (0) of IDC in an image patch. |

---

## 🛠️ Prerequisites and Setup

### 📦 Data Requirement

This notebook requires the **Breast Histopathology Images** dataset (e.g., from Kaggle). The file structure must be as follows:
***Note:*** *The script assumes the image data is accessible in a directory named `train/` or similar structure for `flow_from_directory`.*

### 🖥️ Requirements

This pipeline requires a computational environment capable of handling deep learning tasks, preferably with **GPU acceleration**. You need the following Python libraries installed:

* `tensorflow` / `keras`
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn` (sklearn)

### ⚙️ Execution

1.  **Download** the `Breast_HistoPathology_Images_Analysis_with_CNN_Model_[_Kaggle_=_Dataset].ipynb` file.
2.  **Upload the image dataset** to your running environment (e.g., Google Drive if using Colab).
3.  **Ensure image paths** in the notebook are correctly set up to point to your data folders.
4.  **Execute** all cells sequentially.

---

## 📊 Expected Output

The notebook generates the following critical plots and metrics:

| Output | Description |
| :--- | :--- |
| **Accuracy/Loss Plot** | Line plots showing training and validation accuracy/loss over epochs (used to diagnose overfitting). |
| **Confusion Matrix** | Heatmap visualizing True Positives, True Negatives, False Positives, and False Negatives on the test set. |
| **Classification Report** | Console output summarizing Precision, Recall, and F1-Score for each class. |
| **ROC Curve / AUC** | Graphical representation of model performance across classification thresholds, yielding the Area Under the Curve (AUC) score. |
