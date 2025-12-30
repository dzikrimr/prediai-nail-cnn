# 🤖 AI Diabetes Detection Through Nail Analysis

<strong>Deep Learning-based System for Early Diabetes Risk Detection Using Nail Images</strong>

This project uses Deep Learning technology to detect diabetes risk through nail image analysis. The model uses MobileNetV2 architecture with transfer learning to classify nails into two categories: **prediabetic** and **non-diabetic**.

## 🎯 Project Description

<p align="center">
<strong>AI System for Early Diabetes Detection</strong>
</p>

This AI system was developed to help with early detection of diabetes risk through visual nail analysis. The model is trained using nail image datasets and can identify patterns associated with prediabetic conditions.

### Key Features:
- ✅ Automatic diabetes risk detection through nail images
- ✅ High accuracy using MobileNetV2
- ✅ Risk analysis with confidence scores
- ✅ Comprehensive evaluation result visualizations
- ✅ Easy-to-use API for integration

## 🏗️ Model Architecture

<p align="center">
<strong>MobileNetV2 with Transfer Learning</strong>
</p>

### Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 224x224 pixels
- **Color Mode**: RGB (3 channels)
- **Transfer Learning**: Frozen base model for feature extraction
- **Custom Layers**: 
  - Global Average Pooling
  - Dense layer (128 units, ReLU activation)
  - Dropout (0.5)
  - Output layer (1 unit, Sigmoid activation)

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Batch Size**: 32
- **Epochs**: 15
- **Data Augmentation**: Rotation, zoom, horizontal flip

## 📁 Project Structure

<p align="center">
<strong>Complete Project Organization</strong>
</p>

```
hand-health-ai/
├── app.py                              # Main application (empty - placeholder)
├── requirements.txt                     # Python dependencies
├── summary.py                          # Dataset summary script
├── README.md                           # Project documentation
├── .gitignore                          # Git ignore rules
│
├── src/                               # Main source code
│   ├── train.py                       # Model training script
│   ├── predict.py                     # Basic prediction script
│   ├── evaluate.py                    # Model evaluation script
│   ├── report.py                      # Confusion matrix report script
│   └── split.py                       # Data splitting script
│
├── models/                            # Trained models
│   └── kuku_model.h5                  # Trained MobileNetV2 model
│
├── testing/                           # Testing files and examples
│   ├── cobakukunormal.jpg             # Example of normal nail image
│   └── predict_risk.py                # Detailed risk prediction script
│
└── data/                              # Dataset (not included in git)
    ├── train/                         # Training data
    ├── valid/                         # Validation data
    └── test/                          # Testing data
```

## 🚀 Installation

<p align="center">
<strong>Setup and Dependencies</strong>
</p>

### Prerequisites
- Python 3.7+
- pip package manager

### Steps

1. **Clone repository**
```bash
git clone <repository-url>
cd hand-health-ai
```

2. **Create virtual environment (optional)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies
```
tensorflow>=2.0
numpy
Pillow
matplotlib
seaborn
scikit-learn
```

## 🎯 Usage

<p align="center">
<strong>How to Use the System</strong>
</p>

### 1. Model Training
To train the model from scratch:
```bash
python src/train.py
```

### 2. Model Evaluation
To evaluate model performance:
```bash
python src/evaluate.py
```

### 3. Basic Prediction
For simple prediction:
```bash
python src/predict.py path/to/your/image.jpg
```

### 4. Detailed Risk Prediction
For more in-depth risk analysis:
```bash
python testing/predict_risk.py path/to/your/image.jpg
```

### 5. Dataset Summary
To view dataset statistics:
```bash
python summary.py
```

## 📊 Evaluation Results

<p align="center">
<strong>Model Performance and Visualizations</strong>
</p>

### Confusion Matrix
![Confusion Matrix](evaluation_confusion_matrix_kuku.png)

*Confusion matrix visualization showing model classification performance on test set.*

### Metrics Visualization
![Evaluation Metrics](evaluation_metrics_bar_kuku.png)

*Comparison chart of Precision, Recall, and F1-score for each class*

### Testing Example Image
![Example Normal Nail](testing/cobakukunormal.jpg)

*Example nail image used for system testing*

### Prediction Categories
- **non_diabet**: Normal nail without diabetes indicators
- **prediabet**: Nail with diabetes risk indicators

### Risk Levels
- **Not Identified**: For non-diabetic predictions
- **Moderate Risk**: Confidence ≤ 70% for prediabetic predictions
- **High Risk**: Confidence > 70% for prediabetic predictions

## 🔧 Prediction API

<p align="center">
<strong>Programming Interface</strong>
</p>

### Predict.py Script
```python
python src/predict.py image_path.jpg
```

**Output:**
```
Prediction: prediabet/non_diabet (XX.XX% confidence)
```

### Predict Risk.py Script
```python
python testing/predict_risk.py image_path.jpg
```

**Output:**
```
Prediction: prediabet (XX.XX% confidence)
Risk Level: [Moderate/High Risk]
Detected Features:
- Uneven texture
- Light color changes
- Slightly wavy surface
```

## 📁 Data Structure

<p align="center">
<strong>Dataset Organization Requirements</strong>
</p>

Dataset should be organized in the following structure:
```
data/
├── train/
│   ├── non_diabet/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── prediabet/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── valid/
│   ├── non_diabet/
│   └── prediabet/
└── test/
    ├── non_diabet/
    └── prediabet/
```

### Data Classification
- **non_diabet**: Normal nail images
- **prediabet**: Nail images with diabetes risk indicators

## 🔬 Methodology

<p align="center">
<strong>Technical Approach</strong>
</p>

### 1. Data Preprocessing
- Resize images to 224x224 pixels
- Normalize pixel values (0-1)
- Data augmentation for training set

### 2. Transfer Learning
- Using pre-trained MobileNetV2
- Frozen base model for feature extraction
- Custom classifier layers

### 3. Training Process
- Data split: train/validation/test
- Early stopping based on validation loss
- Model checkpointing

### 4. Evaluation
- Classification report
- Confusion matrix
- Precision, Recall, F1-score
- Overall accuracy

## 📈 Model Performance

<p align="center">
<strong>Results and Accuracy Metrics</strong>
</p>

The model achieves optimal performance with:
- **High Accuracy**: High accuracy on test set
- **Balanced Performance**: Balanced performance for both classes
- **Low False Positive Rate**: Minimal false positives for diabetes detection

*For detailed metrics, run `python src/evaluate.py`*

## 📝 Important Notes

<p align="center">
<strong>Medical Disclaimer</strong>
</p>

⚠️ **Disclaimer**: This system is for educational and research purposes only. It should not be used as a professional medical diagnostic tool. Always consult with qualified medical professionals for diabetes diagnosis and treatment.

