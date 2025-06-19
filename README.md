# 🧱 Pix2Vox-Inspired 3D Reconstruction Model

This project implements a custom **Pix2Vox-like 3D reconstruction pipeline** using **PyTorch**. It reconstructs 3D voxel grids from multi-view 2D images using a modular design (Encoder → Decoder → Merger → Refiner), optimized for ShapeNet-style datasets. It supports AMP training and includes a Jupyter-based inference notebook.

---

## 🚀 Features

### 🧠 Modular Architecture
- Encoder → Decoder → Merger → Refiner pipeline
- Multi-view voxel reconstruction (default: 32×32×32)
- AMP (`torch.cuda.amp`) support for fast mixed-precision training

### 📦 Dataset Handling
- Custom `ShapeNetDataset` class in `utils/DataSetLoader.py`
- Reads `.binvox` voxel files and multi-view 2D images
- Dataset must be placed inside the root directory as `DataSet/`

### 📓 Jupyter Inference
- `inference_pipeline.ipynb` provided for running predictions
- Visualizes reconstructed voxel grids in 3D

### 📈 Training Utilities
- Real-time progress via `tqdm`
- TensorBoard logging
- Auto-saves checkpoints inside `core/checkpoints/` folder

---

## 🛠️ Tech Stack

- **Framework:** PyTorch  
- **Visualization:** Matplotlib (3D voxels)  
- **AMP Training:** `torch.cuda.amp`  
- **UI (Inference):** Jupyter Notebook  

---

## 📂 Project Structure

```
Pix2Vox/
├── core/
│   ├── train.py              # training script (edit dataset paths here)
│   └── checkpoints/          # gets created automatically
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   ├── merger.py
│   └── refiner.py
├── utils/
│   └── DataSetLoader.py
├── inference_pipeline.ipynb  # notebook for testing predictions
└── DataSet/                  # place your dataset here
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Slayer9966/Pix2Vox.git
cd Pix2Vox
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Dataset

Place your ShapeNet-style dataset as follows:

```
Pix2Vox/
└── DataSet/
    ├── ShapeNetVox32_32/
    └── ShapeNetRendering/
```

Edit the paths manually inside:

```bash
core/train.py
```

---

## 🧠 Pretrained Weights

Download pretrained weights from [Google Drive](https://drive.google.com/file/d/1U1Hr8hPXtdea3P1hwpm2UL_A7efxze_T/view?usp=sharing) and place the `.pth` file in main directory and change the path in inference_pipeline.ipynb:

```bash
model.pth
```



---

## ▶️ Training the Model

```bash
python core/train.py
```

> The `core/checkpoints/` folder will be created automatically when training starts.

---

## 📓 Running Inference

Use the provided notebook and place the images inside the data folder in the main directory so it performs the inference on it:

```bash
inference_pipeline.ipynb
```

- You can set custom view images and run forward passes to visualize voxel outputs.
- Make sure the weights are loaded from the correct checkpoint path.

---

## 📌 Notes

- Training requires CUDA-enabled GPU (recommended)
- Dataset format: `.binvox` voxel + 13-view rendered images
- Logs are saved using `tensorboard`
- Model reconstructs 3D voxel grids from 2D multi-view images

---

## 📜 License

Licensed under the **[MIT License](https://github.com/Slayer9966/2D-TO-3D-multi-view/blob/main/LICENSE)** — free to use, modify, and distribute.

---

## 🙋‍♂️ Author

**Syed Muhammad Faizan Ali**  
📍 Islamabad, Pakistan  
📧 faizandev666@gmail.com  
🔗 [GitHub](https://github.com/Slayer9966) | [LinkedIn](https://www.linkedin.com/in/faizan-ali-7b4275297/)
📢 If you find this project helpful or use it in your work, please consider giving it a ⭐ or letting me know via email or GitHub issues!
