# Face Mask Detection: Supervised Classification & Unsupervised Clustering

A comparative machine learning project exploring **supervised classification** and **unsupervised clustering** of face images with and without masks.

The project evaluates how two handcrafted visual feature representations—**Histogram of Oriented Gradients (HOG)** and **SIFT + Bag of Visual Words (BoVW)**—affect the performance of classical machine learning algorithms.

The experiments were developed as two projects for the **Practical Machine Learning** course.

## Project Overview

The goal is to distinguish between images of people **wearing a face mask (`WithMask`)** and **not wearing a face mask (`WithoutMask`)** through two complementary approaches:

- **Supervised Learning:** Training classifiers using labeled images, comparing Logistic Regression and Support Vector Machines (SVM).
- **Unsupervised Learning:** Grouping images without using their labels during clustering, comparing K-Means and Agglomerative Hierarchical Clustering.

In both approaches, the experiments compare HOG with SIFT/BoVW features.

## Dataset

**[Face Mask Detection ~12K Images Dataset (Kaggle)](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)**

| Split | WithMask | WithoutMask | Total |
|---|---:|---:|---:|
| Training | 5,000 | 5,000 | 10,000 |
| Validation | 400 | 400 | 800 |
| Test | 483 | 509 | 992 |
| **Total** | **5,883** | **5,909** | **11,792** |

The original train/validation/test splits were preserved. Model settings were selected using validation data, and the test split was reserved for final evaluation.

## Feature Extraction

### Histogram of Oriented Gradients (HOG)

- Images were converted to **grayscale** and resized to **64 × 64** pixels.
- HOG captures local edge and shape information using gradient orientations.
- The reported configuration uses **9 orientations**, **8 × 8-pixel cells**, **2 × 2-cell blocks**, and **L2-Hys normalization**, producing a **1,764-dimensional** vector per image.

### SIFT + Bag of Visual Words (BoVW)

- SIFT descriptors were extracted from grayscale images at their original resolution.
- K-Means was used to build a visual vocabulary of **100 words** from a sample of **100,000 training descriptors**.
- Each image was represented as a normalized **100-dimensional histogram** of visual word frequencies.

This comparison tests whether global edge/shape patterns or aggregated local keypoints are more informative for face mask recognition.

## Part 1 — Supervised Classification

Two classification algorithms were evaluated on both feature representations:

- **Logistic Regression:** Tuned regularization strength (`C`) and penalty (`L1`/`L2`).
- **SVM:** Compared linear and RBF kernels, with hyperparameter tuning for `C` and, where applicable, `gamma`.

Feature vectors were standardized using `StandardScaler`, fitted on training data only.

Hyperparameters were chosen using validation accuracy before final evaluation on the test set.

### Test Results

| Model | Feature Representation | Test Accuracy |
|---|---|---:|
| Logistic Regression | **HOG** | **98.09%** |
| Logistic Regression | SIFT/BoVW | 83.37% |
| SVM | **HOG** | **98.09%** |
| SVM | SIFT/BoVW | 92.64% |

**Best result:** Both Logistic Regression and SVM achieved **98.09% test accuracy** using HOG features, each misclassifying only **19 of 992** test images.

## Part 2 — Unsupervised Clustering

Clustering algorithms were fitted without class labels:

- **K-Means:** Tuned the number of clusters and initialization settings.
- **Agglomerative Hierarchical Clustering:** Compared cluster counts, linkage strategies (single, complete, average, Ward), and compatible distance metrics.

Performance was measured with the **Adjusted Rand Index (ARI)**, which compares cluster assignments with true classes while correcting for chance agreement.

Labels were used **for evaluation only**, not for fitting the clusters.

Because Agglomerative Clustering does not provide a built-in `predict()` method for unseen data, clustering was run separately on the validation and test splits with the selected hyperparameters.

### Clustering Results

| Model | Feature Representation | Validation ARI | Test ARI |
|---|---|---:|---:|
| K-Means | HOG | 0.6802 | 0.6863 |
| K-Means | SIFT/BoVW | 0.1541 | 0.1730 |
| Agglomerative Clustering | **HOG** | **0.7963** | **0.7867** |
| Agglomerative Clustering | SIFT/BoVW | 0.1955 | 0.1725 |

**Best result:** Agglomerative Clustering with **Ward linkage, Euclidean distance, two clusters, and HOG features** achieved a **test ARI of 0.7867**.

Inspection of representative and misclustered images suggested that image blur, pixelation, tight cropping, and additional edges around faces could affect cluster assignments.

## Key Findings

1. **Feature representation mattered substantially:** HOG outperformed SIFT/BoVW across both supervised and unsupervised experiments.

2. **Classical supervised methods performed strongly:** Logistic Regression and SVM both reached **98.09% test accuracy** with HOG features.

3. **Unsupervised learning recovered meaningful structure:** Hierarchical Clustering achieved **0.7867 test ARI** without class labels during clustering.

4. **The learning setting changes the interpretation of results:** Classification accuracy and ARI measure different objectives and should not be compared as though they were the same metric.

## Technologies & Methods

**Python · scikit-learn · OpenCV/SIFT · HOG · Logistic Regression · SVM · K-Means · Agglomerative Clustering · Hyperparameter Tuning · Image Preprocessing**

## Documentation

The project was documented in two reports:

- **Project 1:** *Supervised Classification of Face Mask Images using Logistic Regression and SVM*
- **Project 2:** *Face Mask Images Clustering using K-Means and Agglomerative Hierarchical Clustering*

The reports contain additional implementation details, hyperparameter-tuning plots, confusion matrices, and examples of clustered images.

## Notes

- The dataset is publicly available on Kaggle.
- Reported results reflect the experimental settings and original dataset splits described above.
- Comparisons against external studies with different splits or preprocessing are not necessarily directly equivalent.
