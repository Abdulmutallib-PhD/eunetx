# UNetX: Automated Brain Tumor Segmentation in CT Medical Images

## Overview

This project presents **UNetX**, a novel deep learning architecture inspired by UNet++ for automated segmentation of brain tumors from CT medical images.  
The model improves segmentation accuracy, computational efficiency, and generalizability over existing models like UNet++, TransUNet, Swin-PANet, and MedT.

UNetX integrates extended skip connections and a lightweight feature fusion (LFF) mechanism to better capture complex tumor boundaries with high efficiency.

---

## Motivation

Manual segmentation of medical images by clinicians is labor-intensive, time-consuming, and prone to observer variability.  
This project addresses these challenges by automating segmentation using a robust deep learning model trained and evaluated on a curated brain tumor CT dataset.

---

## Key Features

- Custom CNN architecture based on UNet++
- Extended skip connections & lightweight feature fusion
- Trained on **16,000 CT datasets**
- Outperforms baseline models in DSC, IoU, and computational efficiency
- Supports stratified k-fold cross-validation

---

## Results

| Metric                        | UNetX Achieved |
|-------------------------------|-----------------|
| Dice Similarity Coefficient   | **93.04%**      |
| Jaccard Index (IoU)           | **87.05%**      |
| Computational Efficiency      | High            |
| Generalizability              | Verified        |

---


## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10
- NumPy
- OpenCV
- pydicom
- tqdm
- Matplotlib
- Docker (optional for deployment)

Install dependencies:
```bash
pip install -r requirements.txt

Visualization
Segmentation results and metrics are saved in the results/ folder.

```

## References

Research Baseline Models:

- Ronneberger et al., 2015 — *U-Net*
- Zhou et al., 2018 — *UNet++*
- Liao et al., 2022 — *Swin-PANet*
- Maria et al., 2021 — *MedT*
- *This thesis*: **Development and Evaluation of a Custom Deep Learning Architecture (UNetX) for Automated Brain Tumor Segmentation in CT Medical Images** by Alhassan Abdulmutallib (2025)

---

## 👩🏽‍💻 Author

**Alhassan Abdulmutallib**  
Doctor of Philosophy (PhD), Computer Science  
Department of Computer Science  
Baze University, Abuja, Nigeria  

- 🌐 [LinkedIn](https://www.linkedin.com/in/alhassan-abdulmutallib-47381294/)
- 🌐 [Facebook](https://www.facebook.com/people/Alhassan-Abdulmutallib/61554448546375/)
