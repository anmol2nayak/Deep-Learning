# Deep-Learning  
## 🌿 ConvNeXt-Tiny for Crop Disease Detection

### Project Overview  
This sub-project applies deep learning to diagnose crop diseases from leaf images. The goal: offer a fast, automated and reliable tool for farmers, enabling early intervention and improved yields.

### Key Features  
- **State-of-the-Art Architecture:** Uses **ConvNeXt-Tiny**, a modern CNN architecture that delivers strong image classification performance.  
- **Transfer Learning:** Leverages a pretrained model (ImageNet) and fine-tunes it for specific crop disease classes.  
- **High Accuracy:** Achieved near-perfect classification performance on the test dataset in preliminary runs.  
- **Disease Classes Detected:**  
  - `bacterial_blight`  
  - `curl` / `curl_virus`  
  - `fussarium_wilt`  
  - `healthy`

### Methodology  
1. **Data Preparation:** Resized images to `(224, 224)` and normalized using ImageNet mean & standard deviation.  
2. **Model Configuration:** Loaded pretrained `convnext_tiny`, replaced its final layer to output 4 disease classes.  
3. **Training:** Model trained for ~10 epochs using the Adam optimizer (`lr = 0.0001`) and CrossEntropyLoss.

### Installation & Setup  
**Prerequisites:** Python (>= 3.7) and PyTorch.

```bash
git clone https://github.com/anmol2nayak/Deep-Learning
cd Deep-Learning
````

**Install dependencies:** (assuming a virtual environment)

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn
```

**Data structure:**
Ensure your dataset is arranged like:

```
data_dir/
  ├── train/
  │    ├── bacterial_blight/
  │    ├── curl/
  │    ├── fussarium_wilt/
  │    └── healthy/
  ├── val/
  └── test/
```

**Run notebook/training:**
Open and run `Convnext.ipynb` (or your training script) to replicate training & testing.

### Results

The model demonstrated strong generalization; here are key metrics (weighted average) on the test set:

| Run                        | Precision | Recall | F1-Score |
| -------------------------- | --------- | ------ | -------- |
| **Run 1 (Real_Data)**      | 0.94      | 0.93   | 0.93     |
| **Run 2 (Data_Augmented)** | 1.00      | 1.00   | 1.00     |

### Future Work

* Deploy as a **mobile application** to allow real-time diagnostics in the field.
* Expand the dataset to cover **additional crop diseases** and varied field conditions (lighting, occlusion, etc.).
* Integrate **object detection** to localize the diseased region on the leaf (not just classify).
* Incorporate segmentation to identify disease-affected areas and quantify severity.

---

## 🛠 System Architecture Overview

1. **Line Segmentation Module:** Uses a UNet model to identify drone flight path lines (on facade/roof/landing area).
2. **Window & Obstacle Detection Module:** Runs a detector (YOLOv8 / Faster-R-CNN) to find target windows and obstacles in the flight path.
3. **Navigation & Control Module:** A PID controller receives segmentation + detection outputs, computes drone commands to correct drift, maintain heading, avoid collisions and deliver packages.
4. **Simulation / Real-World Integration:** The system was developed and tested using AirSim (Microsoft’s drone/vehicle simulation) and transferred to real-world hardware (flight controller, onboard camera) for field trials.

---

## 📁 Directory Structure

```
Deep-Learning/
  ├── drone_delivery/                # Code for autonomous drone delivery
      ├── segmentation/              # UNet model and training scripts
      ├── detection/                 # YOLOv8 / Faster-R-CNN scripts
      ├── control/                   # PID controller code
      └── simulation/                # AirSim integration, flight tests
  ├── crop_disease_detection/        # Code & notebook for crop disease classification
      ├── data/
      ├── notebooks/
      ├── models/
      └── utils/
  ├── README.md                      # This file
  └── requirements.txt               # Python dependencies
```

---

## ✅ How to Use

1. Clone the repository.
2. Choose the module you want to explore (e.g., `crop_disease_detection/` or `drone_delivery/`).
3. Follow the instructions in the module’s folder (data setup, training, evaluation).
4. For drone delivery, first run simulation in AirSim; once validated, adapt for real-hardware.

---

## 👤 Author

**Anmol Nayak** – Student & creative enthusiast with a passion for art, fashion and learning new tech.
(Feel free to link your GitHub profile, LinkedIn, etc.)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

