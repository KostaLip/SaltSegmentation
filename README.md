# Salt Segmentation with U-Net

A deep learning image segmentation project for identifying salt deposits in seismic images using U-Net architecture. This project demonstrates advanced computer vision techniques including semantic segmentation, data augmentation, and custom metrics implementation.

## Project Overview

Salt segmentation is a specialized application of image segmentation used in seismology to identify salt layers in seismic survey images. This project implements a U-Net convolutional neural network to automatically generate binary masks that distinguish salt deposits from surrounding geological formations.

**Key Achievement:** Achieved 90% Dice coefficient and 82% IoU on validation set, demonstrating highly accurate salt boundary detection in seismic imagery.

## Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **Python** - Primary programming language
- **OpenCV** - Image processing and preprocessing
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization and results plotting
- **scikit-learn** - Dataset splitting and preprocessing
- **Google Colab** - Development and training environment

## Problem Statement

**Input:** Seismic survey images (grayscale, 128x128 pixels)  
**Output:** Binary segmentation masks where:
- 1 (white) = Salt deposit present
- 0 (black) = No salt deposit

**Challenge:** Detect complex patterns and textures in low-quality seismic data with limited training samples.

## Dataset

**Source:** Seismic imaging dataset with paired images and ground truth masks

**Composition:**
- Training set: 3,480 image-mask pairs (87%)
- Validation set: 400 image-mask pairs (10%)
- Evaluation set: 120 image-mask pairs (3%)

**Data Characteristics:**
- Image size: 128x128 pixels (resized from original)
- Format: Grayscale PNG images
- Challenge: Dataset contains many images without salt (empty masks)

## Model Architecture

### U-Net Design

U-Net is the industry-standard architecture for image segmentation, consisting of two main components:

**Encoder (Contracting Path):**
- Captures global context through downsampling
- 3 convolutional blocks with MaxPooling
- Feature maps: 64 → 128 → 256 channels
- Learns high-level patterns and textures

**Bottleneck:**
- Deepest layer with 512 feature maps
- Captures most abstract representations

**Decoder (Expanding Path):**
- Reconstructs spatial resolution through upsampling
- 3 deconvolutional blocks with Conv2DTranspose
- Skip connections from encoder preserve fine details
- Channels: 256 → 128 → 64

**Final Layer:**
- 1x1 convolution with sigmoid activation
- Outputs probability map (0-1) for each pixel

### Architecture Details

```python
Input: (128, 128, 1)
├── Encoder Block 1: Conv2D(64) → Conv2D(64) → MaxPool
├── Encoder Block 2: Conv2D(128) → Conv2D(128) → MaxPool
├── Encoder Block 3: Conv2D(256) → Conv2D(256) → MaxPool
├── Bottleneck: Conv2D(512) → Conv2D(512)
├── Decoder Block 1: UpSample(256) + Skip Connection → Conv2D(256) → Conv2D(256)
├── Decoder Block 2: UpSample(128) + Skip Connection → Conv2D(128) → Conv2D(128)
├── Decoder Block 3: UpSample(64) + Skip Connection → Conv2D(64) → Conv2D(64)
└── Output: Conv2D(1, sigmoid) → (128, 128, 1)
```

## Key Features

### 1. Data Preprocessing

**Image Normalization:**
- Pixel values scaled to [0, 1] range
- Improves neural network convergence
- Reduces training time

**Mask Binarization:**
- All non-zero pixels converted to 1
- Ensures binary classification per pixel
- Added channel dimension for consistency

**Sorted Pairing:**
- Images and masks sorted by filename
- Guarantees correct image-mask correspondence

### 2. Data Augmentation

Applied real-time augmentation to increase dataset diversity and prevent overfitting:

- **Rotation:** ±10 degrees
- **Width/Height Shift:** ±10%
- **Zoom:** ±10%
- **Horizontal Flip:** Random left-right flipping

**Batch Processing:**
- Batch size: 32 images per iteration
- Synchronized augmentation for images and masks
- Fixed random seed for reproducibility

### 3. Custom Metrics

#### IoU (Intersection over Union)
```python
IoU = (Prediction ∩ Ground Truth) / (Prediction ∪ Ground Truth)
```
Measures overlap between predicted and actual salt regions. Final validation IoU: **82.6%**

#### Dice Coefficient
```python
Dice = 2 × (Prediction ∩ Ground Truth) / (Prediction + Ground Truth)
```
Similar to IoU but emphasizes overlap more. Final validation Dice: **90.0%**

**Why these metrics?**
- Accuracy is misleading for imbalanced segmentation
- IoU and Dice directly measure segmentation quality
- Industry-standard metrics for medical/geological imaging

### 4. Training Strategy

**Optimizer:** Adam (learning_rate=1e-4)
- Adaptive learning rate for each parameter
- Faster convergence than standard SGD

**Loss Function:** Binary Crossentropy
- Appropriate for binary classification per pixel
- Penalizes incorrect pixel predictions

**Callbacks:**
- **EarlyStopping:** Stops training if no improvement for 10 epochs
- **ReduceLROnPlateau:** Reduces learning rate by 10x if stuck (patience=5)
- **ModelCheckpoint:** Saves best model based on validation Dice coefficient

**Training Duration:** 50 epochs (early stopping at epoch 49)

## Results

### Training Metrics

| Metric | Training | Validation |
|--------|----------|------------|
| **Accuracy** | 93.7% | 95.8% |
| **Loss** | 0.151 | 0.115 |
| **IoU** | 77.4% | 82.6% |
| **Dice** | 87.0% | 90.0% |

### Key Observations

**Learning Progression:**
- First epochs: IoU and Dice very low (~3-4%) as network initializes
- Epoch 3-10: Rapid improvement in all metrics
- Epoch 10-30: Steady learning with occasional plateaus
- Epoch 30+: Fine-tuning with reduced learning rate
- Epoch 49: Best performance achieved

**Generalization:**
- Validation metrics consistently high throughout training
- No significant overfitting observed
- Validation Dice slightly higher than training due to augmentation

**Model Stability:**
- Loss decreased consistently without erratic fluctuations
- Metrics improved smoothly across epochs
- Early stopping prevented unnecessary training

## Implementation Highlights

### Transparent Donut Handling
The U-Net architecture naturally handles complex shapes including:
- Internal boundaries
- Transparent regions
- Multiple disconnected salt deposits

### Skip Connections
Critical for preserving fine-grained details:
- Encoder features concatenated with decoder features
- Enables precise boundary detection
- Prevents information loss during downsampling

### Reproducibility
Fixed random seeds across all libraries:
```python
seed = 2019
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
```

Ensures consistent results across runs for:
- Dataset splitting
- Data augmentation
- Weight initialization

## Getting Started

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy, Matplotlib, scikit-learn

### Running on Google Colab

1. **Upload notebook**
   ```
   Upload SaltSegmentation.ipynb to Google Colab
   ```

2. **Upload training data**
   ```python
   # Upload train.zip to Colab
   # Notebook automatically extracts and processes data
   ```

3. **Execute cells sequentially**
   - Data preprocessing
   - Model definition
   - Training
   - Evaluation

## Usage

### Training the Model

```python
# Load and preprocess data
x = getAllImages()
y = getAllMasks()

# Split dataset
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.13)
xVal, xEval, yVal, yEval = train_test_split(xTest, yTest, test_size=0.23)

# Create model
model = unetModel(inputSize=(128, 128, 1))

# Train with augmentation
history = model.fit(
    trainGenerator(batch_size=32),
    epochs=50,
    validation_data=(xVal, yVal),
    callbacks=callbacks
)
```

### Making Predictions

```python
# Load best model
model = load_model('bestModel.keras', 
                   custom_objects={
                       "dice_coefficient": dice_coefficient,
                       "binary_iou_metric": binary_iou_metric
                   })

# Predict on new images
predictions = model.predict(new_images)
masks = (predictions > 0.5).astype(np.uint8)
```

## Visualizations

The notebook generates comprehensive visualizations:
- Loss curves (training vs validation)
- Accuracy progression
- IoU and Dice metrics over epochs
- Side-by-side comparison: Original Image | Ground Truth | Prediction

## Challenges & Solutions

### Challenge 1: Limited Dataset
**Problem:** Only ~4,000 images with many duplicates  
**Solution:** Aggressive data augmentation to artificially expand dataset

### Challenge 2: Empty Masks
**Problem:** Many images contain no salt deposits  
**Solution:** Balanced training with proper metric selection (IoU/Dice handle this well)

### Challenge 3: Low Image Quality
**Problem:** Seismic images are noisy and low contrast  
**Solution:** U-Net's skip connections preserve fine details during reconstruction

### Challenge 4: Small Details
**Problem:** Salt boundaries can be very fine-grained  
**Solution:** Multi-scale feature learning through encoder-decoder architecture

## Future Enhancements

- **Larger Dataset:** Incorporate additional seismic survey data
- **Test Set with Masks:** Enable quantitative evaluation on unseen data
- **Advanced Augmentation:** Elastic deformations, intensity variations
- **Ensemble Models:** Combine multiple U-Net variants for better predictions
- **Post-processing:** Morphological operations to clean up predictions
- **Transfer Learning:** Pre-train on natural images before seismic data
- **3D Segmentation:** Extend to volumetric seismic data

## Key Learning Points

- **U-Net Architecture:** Implemented industry-standard segmentation network
- **Custom Metrics:** Created IoU and Dice coefficient from scratch
- **Data Augmentation:** Applied real-time augmentation for limited datasets
- **Binary Segmentation:** Solved pixel-wise classification problem
- **Callback Strategies:** Used EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Reproducibility:** Fixed random seeds for consistent experiments
- **Domain Adaptation:** Applied computer vision to geological/seismic imaging

## Academic Context

This project was developed for the **Pattern Recognition and Presentation** course at the Faculty of Technical Sciences, University of Novi Sad (2024/2025).

**Course Topics Covered:**
- Image segmentation fundamentals
- Convolutional neural networks
- U-Net architecture
- Semantic segmentation
- Custom loss functions and metrics
- Data augmentation techniques

## Applications

Salt segmentation has real-world applications in:
- **Oil & Gas Exploration:** Identifying salt domes for drilling sites
- **Geological Surveys:** Understanding subsurface structures
- **Seismic Analysis:** Interpreting underground formations
- **Risk Assessment:** Evaluating drilling hazards
