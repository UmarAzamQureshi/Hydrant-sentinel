# Hydrant-Sentinel: AI-Powered Fire Hydrant Blockage Detection

## 🔥 Problem Statement

Fire hydrants are critical emergency infrastructure that must remain accessible at all times. However, parked vehicles often block hydrants, creating dangerous delays during fire emergencies. This can result in:
- Delayed emergency response times
- Loss of property and lives due to inaccessible water sources
- Violations of fire safety regulations
- Fines and penalties for blocking fire hydrants

Hydrant-Sentinel uses computer vision and machine learning to automatically detect when vehicles are illegally parked too close to fire hydrants, enabling authorities to take immediate action.

## 🎯 Solution Overview

Our AI system combines:
- **YOLOv8 Object Detection**: Identifies fire hydrants and vehicles in real-time
- **Smart Distance Analysis**: Calculates spatial relationships between hydrants and vehicles
- **Blockage Detection Logic**: Determines if a vehicle is illegally blocking a fire hydrant
- **Automated Alerting**: Provides immediate feedback on blockage violations

## 🏗️ Architecture & Model

### Model Architecture
- **Base Model**: YOLOv8n (YOLO version 8, nano variant)
- **Training Classes**: 2 classes (hydrant, car)
- **Input Size**: 640x640 pixels
- **Training Epochs**: 50 epochs
- **Optimizer**: Auto (Adam optimizer)
- **Data Augmentation**: Flip, HSV color variation, mosaic blending

### Model Performance
The model is trained on a custom dataset containing:
- **Training Images**: 4,369 images
- **Validation Images**: 1,093 images  
- **Total Dataset**: 5,462 annotated images
- **Object Detection**: Real-time detection of hydrants and cars

### Blockage Detection Logic
The system uses sophisticated algorithms to determine blockage:

1. **Spatial Buffer Analysis**: Creates an expanded safety zone around each hydrant (default: 1.6x horizontal, 1.8x vertical expansion)
2. **Intersection Detection**: Checks if any vehicle bounding boxes intersect with the enlarged hydrant safety zone
3. **Proximity Analysis**: Analyzes center-to-center distances relative to hydrant size
4. **Configurable Thresholds**: Adjustable parameters for different environment requirements

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.x**
- **YOLOv8/Ultralytics**: State-of-the-art object detection
- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computations
- **scikit-learn**: Data splitting and ML utilities
- **PyTorch**: Deep learning framework (automatically installed with YOLOv8)

### Dependencies
```python
ultralytics==8.3.203
opencv-python==4.12.0.88
numpy==2.2.6
scikit-learn==1.7.2
torch==2.8.0
torchvision==0.23.0
PIL==11.3.0
matplotlib==3.10.6
```

## 📁 Project Structure

```
Hydrant-sentinel/
├── Hydrant/                 # Hydrant training images
├── Car/                     # Car training images  
├── hydrant_dataset/         # Organized training dataset
│   ├── images/
│   │   ├── train/          # Training images (4,369 files)
│   │   └── val/            # Validation images (1,093 files)
│   └── labels/
│       ├── train/          # Training labels
│       └── val/            # Validation labels
├── script/                 # Core automation scripts
│   ├── auto_label_hydrants.py    # Automated hydrant labeling
│   ├── auto_label_car.py         # Automated car labeling  
│   ├── splitdata.py              # Dataset organization
│   ├── blockage_detected.py      # Main detection script
│   └── yolov8n.pt                # Pre-trained weights
├── runs/                   # Training results
│   └── detect/train/
│       └── weights/
│           ├── best.pt         # Best trained weights
│           └── last.pt         # Last training weights
├── hydrant_data.yaml       # Dataset configuration
└── venv/                   # Python virtual environment
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Windows 10/11 or Linux
- 8GB+ RAM recommended
- NVIDIA GPU (optional, for faster training)

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Hydrant-sentinel.git
cd Hydrant-sentinel
```

#### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install scikit-learn
pip install torch torchvision
pip install pillow
pip install matplotlib
```

#### 4. Verify Installation
```bash
python -c "from ultralytics import YOLO; print('Installation successful!')"
```

## 🎯 Usage Instructions

### Workflow Overview

**Step 1: Data Preparation**
```bash
# Auto-label hydrants in your images (Class 0)
python script/auto_label_hydrants.py

# Auto-label cars in your images (Class 1)  
python script/auto_label_car.py

# Organize data into YOLO format
python script/splitdata.py
```

**Step 2: Train the Model**
```bash
yolo train data=hydrant_data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

**Step 3: Detect Blockages**
```bash
python script/blockage_detected.py path/to/your/images
```

### Detailed Commands

#### 1. Prepare Training Data
```bash
# Label hydrants in your image dataset
python script/auto_label_hydrants.py

# Label cars in your image dataset  
python script/auto_label_car.py

# Split data into train/validation sets
python script/splitdata.py
```

#### 2. Train the YOLO Model
```bash
# Start training with custom parameters
yolo train \
  data=hydrant_data.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  device=cpu  # or '0' for GPU
  
# Resume training from checkpoint
yolo train \
  resume=runs/detect/train/weights/last.pt
```

#### 3. Run Blockage Detection
```bash
# Detect blockages in single image
python script/blockage_detected.py image.jpg

# Detect blockages in directory
python script/blockage_detected.py /path/to/images/

# Custom confidence threshold
python script/blockage_detected.py images/ --conf 0.5

# Custom model weights  
python script/blockage_detected.py images/ --model runs/detect/train/weights/best.pt

# Adjust blockage detection parameters
python script/blockage_detected.py images/ --bufx 2.0 --bufy 2.5 --center_factor 0.7
```

#### 4. Advanced Options
```bash
# Comprehensive detection with all options
python script/blockage_detected.py \
  --source image.jpg \
  --model runs/detect/train/weights/best.pt \
  --conf 0.25 \
  --bufx 1.6 \
  --bufy 1.8 \
  --center_factor 0.8 \
  --save_dir results/
```

### Output Files
- **Detection Results**: Saved to `runs/blockage/` by default
- **Annotated Images**: Images with bounding boxes and "BLOCKAGE DETECTED" labels
- **Confidence Scores**: Displayed for each detected object
- **Status Reports**: Console output showing detection statistics

## ⚙️ Configuration Parameters

### Training Parameters
- **Epochs**: 50 (adjustable based on dataset size)
- **Batch Size**: 16 (adjust based on GPU memory)
- **Image Size**: 640x640 (YOLO standard)
- **Confidence Threshold**: 0.25 (minimum confidence for detection)

### Detection Parameters
- **Buffer Scaling**:
  - `bufx`: Horizontal expansion factor (default: 1.6)
  - `bufy`: Vertical expansion factor (default: 1.8)
- **Proximity Factor**: 
  - `center_factor`: Distance threshold (default: 0.8)

### Model Modifications
Edit `hydrant_data.yaml` to customize:
- Training/validation paths
- Number of classes
- Class names
- Dataset configuration

## 🔧 Troubleshooting

### Common Issues

**1. "Module not found" errors**
```bash
# Reinstall missing packages
pip install --upgrade ultralytics opencv-python numpy
```

**2. CUDA/GPU Issues**
```bash
# Install CPU-only PyTorch if GPU causes issues
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**3. Memory Errors**
- Reduce batch size in training: `batch=8`
- Resize images: `imgsz=416`
- Process images in smaller batches

**4. Training Performance**
- Monitor GPU usage: `nvidia-smi`
- Check train/val loss convergence
- Increase epochs if loss plateaus

### Performance Optimization
- Use GPU for training (`device=0`)
- Adjust batch size based on available RAM
- Use smaller image sizes for faster processing
- Cache dataset for faster training

## 📊 Model Performance

The trained model achieves high accuracy in detecting hydrants and vehicles with real-time performance on both CPU and GPU. The blockage detection algorithm provides reliable classification between blocked and unblocked hydrants.

## 🌟 Real-World Applications

- **Municipal Fire Departments**: Automated monitoring of hydrant accessibility
- **Traffic Management**: Validation of parking violations near fire hydrants  
- **Smart City Infrastructure**: Integration with existing surveillance systems
- **Emergency Services**: Real-time hydrant status for emergency responders
- **Code Enforcement**: Automated detection of fire safety violations

## 🔮 Future Enhancements

- Real-time video stream processing
- Integration with existing parking enforcement systems
- Mobile app for field personnel
- Database logging of violations
- Advanced alerting system with GIS integration

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

---

**Hydrant-Sentinel**: Keeping emergency infrastructure accessible through AI-powered monitoring 🚒🔥💧
