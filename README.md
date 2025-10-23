# 🌿 ConvNeXt-Tiny for Crop Disease Detection

## Project Overview
This project leverages deep learning to accurately classify common crop diseases from leaf images. The goal is to provide a fast, automated, and reliable diagnostic tool for farmers, helping to ensure early intervention and improve crop yield.

## Key Features
* **State-of-the-Art Architecture:** Uses **ConvNeXt-Tiny**, a modern Convolutional Neural Network (CNN) architecture known for high performance in image classification.
* **Transfer Learning:** Utilizes a pretrained model (on ImageNet) and fine-tunes it for superior performance on a specific crop disease dataset.
* **High Accuracy:** Achieved near-perfect classification performance on the test dataset (up to 100% test accuracy in one run).
* **Disease Classes Detected:**
    * `bacterial_blight`
    * `curl` / `curl_virus`
    * `fussarium_wilt`
    * `healthy`

## Methodology
1.  **Data Preparation:** Images were resized to `(224, 224)` and normalized using standard ImageNet means and standard deviations.
2.  **Model Configuration:** The pretrained `convnext_tiny` model was loaded, and its final classification layer was adjusted to output scores for 4 disease classes.
3.  **Training:** The model was trained for **10 epochs** using the **Adam** optimizer (`lr=0.0001`) and **CrossEntropyLoss**.

## Installation and Setup

This project requires Python and PyTorch.

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPO_URL]
    cd Convnext
    ```

2.  **Install dependencies:**
    *(Assuming you use a virtual environment)*
    ```bash
    pip install torch torchvision numpy scikit-learn matplotlib seaborn
    ```

3.  **Data Structure:** Ensure your dataset is organized in the following structure within the project directory:
    ```
    data_dir/
    ├── train/
    │   ├── bacterial_blight/
    │   ├── curl/
    │   ├── fussarium_wilt/
    │   └── healthy/
    ├── val/
    └── test/
    ```

4.  **Run the notebook:**
    Open and run `Convnext.ipynb` to replicate the training and testing process.

## Results
The model demonstrated strong generalization with the following weighted average metrics on the test set:

| Run | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :--- | :--- | :--- |
| **Run 1 (Real\_Data)** | 0.94 | 0.93 | 0.93 |
| **Run 2 (DATA)** | 1.00 | 1.00 | 1.00 |

## Future Work
* Deployment into a **mobile application** for real-time field diagnostics.
* Expansion of the dataset to include more diseases and variable field conditions.
* Integration of **object detection** to pinpoint the exact location of the disease on the leaf.

---
**Author:** [Your Name]
**License:** [Choose a license, e.g., MIT]
