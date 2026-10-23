from pathlib import Path

data_path = Path("dataset")

for split in ["Train", "Validation", "Test"]:
    split_path = data_path / split
    print(split, ":", [folder.name for folder in split_path.iterdir()])

for split in ["Train", "Validation", "Test"]:
    for class_n in ["WithMask", "WithoutMask"]:
        class_path = data_path / split / class_n
        nr_images = len(list(class_path.iterdir()))
        print(class_n, ":", nr_images)


# check if size of images are different
from PIL import Image

sample_path = data_path / "Train" / "WithMask"

for img_path in list(sample_path.iterdir())[:20]:
    img = Image.open(img_path)
    print(img_path.name, ":", img.size)


import numpy as np

# we have images of different sizes
# need more info

sizes = []
image_data = []

for split in ["Train"]:
    for label in ["WithMask", "WithoutMask"]:
        class_path = data_path / split / label

        for img_path in class_path.iterdir():
            with Image.open(img_path) as img:
                sizes.append(img.size)
                image_data.append((img_path, img.size))

widths = [w for w, h in sizes]
heights = [h for w, h in sizes]

print(f"Training set contains {len(sizes)} images in total")
print(f"Minimum width: {min(widths)}")
print(f"Maximum width: {max(widths)}")
print(f"Median width: {np.median(widths)}")
print(f"Minimum height: {min(heights)}")
print(f"Maximum height: {max(heights)}")
print(f"Median height: {np.median(heights)}")

# we will want same number of HOG features -> we need to resize images
# trying to find the optimal size

smaller_than_64 = 0
smaller_than_128 = 0
at_least_128 = 0

for w,h in sizes:
    if w < 64 or h <64:
        smaller_than_64+=1
    if w < 128 or h < 128:
            smaller_than_128+=1
    else: at_least_128+=1

print(f"Images smaller than 64x64: {smaller_than_64}")
print(f"Images smaller than 128x128: {smaller_than_128}")
print(f"Images at least 128x128: {at_least_128}")


import matplotlib.pyplot as plt

# show examples of images of different sizes
# rescaled to 64x64

examples = [25, 64, 120, 563]

for example in examples:
    best_path = None
    best_size = None
    best_diff = float("inf")

    for img_path, size in image_data:
        w,h = size
        diff = abs(w-example) + abs(h-example)
        if diff < best_diff:
            best_diff = diff
            best_path = img_path
            best_size = size

    img = Image.open(best_path)
    resized_img = img.resize((64, 64))

    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.title(f"Original: {best_size[0]}x{best_size[1]}")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(3, 3))
    plt.imshow(resized_img)
    plt.title(f"Resized: {best_size[0]}x{best_size[1]} -> 64x64")
    plt.axis("off")
    plt.show()

# HOG feature extraction

from skimage.feature import hog

def extract_hog_features(img_path, orientations=9, 
                         pixels_per_cell = (8,8), 
                         cells_per_block = (2,2)):
    # make image grayscale
    img = Image.open(img_path).convert("L")

    img = img.resize((64,64))
    img_array = np.array(img)

    # hog feature vector
    hog_features = hog(img_array, orientations = orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       block_norm ="L2-Hys") # reduce effect of illumination

    return hog_features

# check on an image
img_path = data_path / "Train" /"WithMask"
img = list(img_path.iterdir())[0]
features = extract_hog_features(img)
print(f"Number of features extracted: {len(features)}")
print("First 10 features:", features[:10])


# apply hog feature extraction to all images in a split

def apply_hog_on_split(split_name, orientations=9,
                        pixels_per_cell=(8,8),
                        cells_per_block=(2,2)):
    feature_vectors = []
    real_labels = []
    for class_name in ["WithMask", "WithoutMask"]:
        class_path = data_path / split_name / class_name
        for img_path in class_path.iterdir():
            hog_features = extract_hog_features(img_path,
                                                orientations,
                                                pixels_per_cell,
                                                cells_per_block)
            feature_vectors.append(hog_features)
            real_labels.append(class_name)
    return np.array(feature_vectors), np.array(real_labels)

# apply feature extraction on training and validation sets
train_data_hog, train_labels = apply_hog_on_split("Train")
val_data_hog, val_labels = apply_hog_on_split("Validation")

import cv2

# method 2 for feature extraction: SIFT + BoVW

# trying SIFT on one image first
img_path = data_path / "Train" / "WithMask"
img = list(img_path.iterdir())[0]

# image in grayscale
img_gs = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)

sift_extractor = cv2.SIFT_create()
keypoints, descriptors = sift_extractor.detectAndCompute(img_gs, None)

print(f"Keypoints: {len(keypoints)}")
print(f"Descriptor shape: {descriptors.shape}")

# show image with keypoints

img_keypoints = cv2.drawKeypoints(img_gs, keypoints, None)
plt.figure(figsize=(5,5))
plt.imshow(img_keypoints, cmap="gray")
plt.axis("off")
plt.show()

sift_ext = cv2.SIFT_create()

def extract_sift_features_from_image(img_path):
    img = cv2.imread(str(img_path),cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift_ext.detectAndCompute(img,None)
    return descriptors


# extract sift features from split

def extract_sift_features_from_split(split_name):
    descriptors_list = []
    labels = []

    for class_name in ["WithMask", "WithoutMask"]:
            class_path = data_path / split_name / class_name
            for img_path in class_path.iterdir():
                descriptors = extract_sift_features_from_image(img_path)
                descriptors_list.append(descriptors)
                labels.append(class_name)

    return descriptors_list, np.array(labels)



train_sift_descriptors, train_labels_sift = extract_sift_features_from_split("Train")
print(len(train_sift_descriptors))
print(len(train_labels_sift))


# get all train descriptors
all_train_descriptors = []
for descriptors in train_sift_descriptors:
    if descriptors is not None:
        all_train_descriptors.append(descriptors)

all_train_descriptors = np.vstack(all_train_descriptors)
print(all_train_descriptors.shape)


# we have more than 10^6 descriptors from the training set
# we will choose 10^5 descriptors randomly
# in order to reduce computational cost

np.random.seed(42)

random_desc = np.random.choice(
    len(all_train_descriptors), size = 100000, replace=False )

final_descriptors = all_train_descriptors[random_desc]
print(final_descriptors.shape)


# build the vocabulary using k-means
# one visual word = one centroid
# we try with 100 visual words
# 100 clusters -> 100 centroids -> 100 visual words

from sklearn.cluster import KMeans

vocab_size = 100

vocab_kmeans = KMeans(n_clusters=vocab_size,
                           random_state=42,
                           n_init=10)

vocab_kmeans.fit(final_descriptors)

print(vocab_kmeans.cluster_centers_.shape)


def bovw_features(descriptors):
    bovw = []
    for d in descriptors:
        if d is None:
            histogram = np.zeros(vocab_size)
        else:
            visual_words = vocab_kmeans.predict(d)
            histogram = np.bincount(visual_words, minlength=vocab_size)
            histogram = histogram/histogram.sum()
        bovw.append(histogram)
    return np.array(bovw)


train_bovw = bovw_features(train_sift_descriptors)
print(train_bovw.shape)
print(train_bovw[0])
print(train_bovw[0].sum())


val_sift_descriptors, val_labels_sift = extract_sift_features_from_split("Validation")

print(len(val_sift_descriptors))
print(len(val_labels_sift))


val_bovw = bovw_features(val_sift_descriptors)
print(val_bovw.shape)
print(len(val_labels_sift))


# hyperparameter tuning for LR + HOG features

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

scaler_hog = StandardScaler()
train_data_hog_scaled = scaler_hog.fit_transform(train_data_hog)
val_data_hog_scaled = scaler_hog.transform(val_data_hog)

C_grid = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
penalties = ["l1", "l2"]

results_hog_lr = []
best_acc = 0
best_C = None
best_penalty = None
best_model = None

for penalty in penalties:
    for C in C_grid:
        lr = LogisticRegression(C=C, penalty=penalty, solver="liblinear", max_iter=1000)
        lr.fit(train_data_hog_scaled, train_labels)

        train_pred = lr.predict(train_data_hog_scaled)
        train_acc = accuracy_score(train_labels, train_pred)

        val_pred = lr.predict(val_data_hog_scaled)
        val_acc = accuracy_score(val_labels, val_pred)

        results_hog_lr.append([C,penalty,train_acc,val_acc])

        if val_acc > best_acc:
            best_acc = val_acc
            best_C = C
            best_penalty = penalty
            best_model = lr

#best_train_pred = best_model.predict(train_data_hog_scaled)
best_val_pred = best_model.predict(val_data_hog_scaled)
#best_train_acc = accuracy_score(train_labels, best_train_pred)

print(f"Best C: {best_C}")
print(f"Best penalty: {best_penalty}")
print(f"Best validation accuracy: {best_acc}")

conf_matrix = confusion_matrix(val_labels, best_val_pred)
print("Confusion matrix:")
print(conf_matrix)

show_conf_matrix = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                          display_labels=best_model.classes_)
show_conf_matrix.plot()
plt.show()


import pandas as pd 

results_hog_lr_df = pd.DataFrame(results_hog_lr, columns = ["C","Penalty", "Train Acc", "Validation Acc"])
print(results_hog_lr_df)

for penalty in penalties:
    data = results_hog_lr_df[
        results_hog_lr_df["Penalty"] == penalty
    ]
    plt.plot(data["C"], data["Validation Acc"], marker="o", label=penalty.upper())

plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid()
plt.title("Logistic Regression + HOG Features - Hyperparameter Tuning")
plt.show()



# hyperparameter tuning for LR + SIFT and BoVW features

scaler_bovw = StandardScaler()
train_bovw_scaled = scaler_bovw.fit_transform(train_bovw)
val_bovw_scaled = scaler_bovw.transform(val_bovw)

C_grid = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
penalties = ["l1", "l2"]

results_bovw_lr = []
best_acc_bovw = 0
best_C_bovw = None
best_penalty_bovw = None
best_model_bovw = None

for penalty in penalties:
    for C in C_grid:
        lr = LogisticRegression(C=C, penalty=penalty, solver="liblinear", max_iter=1000)
        lr.fit(train_bovw_scaled, train_labels_sift)

        train_pred = lr.predict(train_bovw_scaled)
        train_acc = accuracy_score(train_labels_sift, train_pred)

        val_pred = lr.predict(val_bovw_scaled)
        val_acc = accuracy_score(val_labels_sift, val_pred)

        results_bovw_lr.append([C,penalty,train_acc,val_acc])

        if val_acc > best_acc_bovw:
            best_acc_bovw = val_acc
            best_C_bovw = C
            best_penalty_bovw = penalty
            best_model_bovw = lr

best_train_pred_bovw = best_model_bovw.predict(train_bovw_scaled)
best_val_pred_bovw = best_model_bovw.predict(val_bovw_scaled)

print(f"Best C: {best_C_bovw}")
print(f"Best penalty: {best_penalty_bovw}")
print(f"Best validation accuracy: {best_acc_bovw}")



conf_matrix_bovw = confusion_matrix(val_labels_sift, best_val_pred_bovw)
print("Confusion matrix:")
print(conf_matrix_bovw)

show_conf_matrix_bovw = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_bovw,
                                          display_labels=best_model_bovw.classes_)
show_conf_matrix_bovw.plot()
plt.show()


results_bovw_lr_df = pd.DataFrame(results_bovw_lr, columns = ["C","Penalty", "Train Acc", "Validation Acc"])
print(results_bovw_lr_df)


for penalty in penalties:
    data = results_bovw_lr_df[
        results_bovw_lr_df["Penalty"] == penalty
    ]
    plt.plot(data["C"], data["Validation Acc"], marker="o", label=penalty.upper())

plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid()
plt.title("Logistic Regression + SIFT/BoVW Features - Hyperparameter Tuning")
plt.show()



# hyperparameter tuning for SVM + HOG features

from sklearn.svm import SVC

C_grid_svm = [0.01,0.1,1,10]
gamma_grid = ["scale", 0.0001, 0.001]

results_hog_svm = []

best_acc_hog_svm = 0
best_kernel_hog_svm = None
best_C_hog_svm = None
best_gamma_hog_svm = None
best_model_hog_svm = None

for C in C_grid_svm:
    svm = SVC(
        C=C,
        kernel = "linear"
    )
    svm.fit(train_data_hog_scaled, train_labels)

    train_pred = svm.predict(train_data_hog_scaled)
    train_acc = accuracy_score(train_labels, train_pred)
    val_pred = svm.predict(val_data_hog_scaled)
    val_acc = accuracy_score(val_labels, val_pred)

    results_hog_svm.append([C,"linear",None,train_acc,val_acc])

    if val_acc > best_acc_hog_svm:
        best_acc_hog_svm = val_acc
        best_C_hog_svm = C
        best_kernel_hog_svm = "linear"
        best_gamma_hog_svm = None
        best_model_hog_svm = svm

for C in C_grid_svm:
    for gamma in gamma_grid:
        svm = SVC(C=C, kernel = "rbf", gamma=gamma)
        svm.fit(train_data_hog_scaled, train_labels)
        train_pred = svm.predict(train_data_hog_scaled)
        train_acc = accuracy_score(train_labels, train_pred)
        val_pred = svm.predict(val_data_hog_scaled)
        val_acc = accuracy_score(val_labels, val_pred)
        results_hog_svm.append([C,"rbf",gamma,train_acc,val_acc])

        if val_acc > best_acc_hog_svm:
                best_acc_hog_svm = val_acc
                best_C_hog_svm = C
                best_kernel_hog_svm = "rbf"
                best_gamma_hog_svm = gamma
                best_model_hog_svm = svm

best_val_pred_hog_svm = best_model_hog_svm.predict(val_data_hog_scaled)



results_hog_svm_df = pd.DataFrame(results_hog_svm, 
                                  columns= ["C","Kernel","Gamma","Train Acc", "Validation Acc"])
print(results_hog_svm_df)


conf_matrix_hog_svm = confusion_matrix(val_labels, best_val_pred_hog_svm)
# trebuie rulat iar codul de sus cu 
# best_val_pred_hog_svm, momentan e rulat fara
print("Confusion matrix:")
print(conf_matrix_hog_svm)

show_conf_matrix_hog_svm = ConfusionMatrixDisplay(
    confusion_matrix = conf_matrix_hog_svm,
    display_labels = best_model_hog_svm.classes_
)
show_conf_matrix_hog_svm.plot()
plt.show()



# graph for linear kernel - hyperparameter C

results_linear = results_hog_svm_df[results_hog_svm_df["Kernel"] == "linear"]
plt.plot(
    results_linear["C"],
    results_linear["Validation Acc"], marker ="o"
)
plt.title("SVM + HOG - Linear Kernel")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Validation Accuracy")
plt.grid()
plt.show()



rbf_results = results_hog_svm_df[results_hog_svm_df["Kernel"] == "rbf"]
for gamma in gamma_grid:
    data = rbf_results[rbf_results["Gamma"] == gamma]
    plt.plot(data["C"], data["Validation Acc"], marker="o", label=f"gamma={gamma}")

plt.title("SVM + HOG - RBF Kernel")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid()
plt.show()




# hyperparameter tuning for SVM + SIFT/BoVW Features

from sklearn.svm import SVC

C_grid_svm = [0.01,0.1,1,10]
gamma_grid = ["scale", 0.0001, 0.001]

results_bovw_svm = []

best_acc_bovw_svm = 0
best_kernel_bovw_svm = None
best_C_bovw_svm = None
best_gamma_bovw_svm = None
best_model_bovw_svm = None

for C in C_grid_svm:
    svm = SVC(
        C=C,
        kernel = "linear"
    )
    svm.fit(train_bovw_scaled, train_labels_sift)

    train_pred = svm.predict(train_bovw_scaled)
    train_acc = accuracy_score(train_labels_sift, train_pred)
    val_pred = svm.predict(val_bovw_scaled)
    val_acc = accuracy_score(val_labels_sift, val_pred)

    results_bovw_svm.append([C,"linear",None,train_acc,val_acc])

    if val_acc > best_acc_bovw_svm:
        best_acc_bovw_svm = val_acc
        best_C_bovw_svm = C
        best_kernel_bovw_svm = "linear"
        best_gamma_bovw_svm = None
        best_model_bovw_svm = svm

for C in C_grid_svm:
    for gamma in gamma_grid:
        svm = SVC(C=C, kernel = "rbf", gamma=gamma)
        svm.fit(train_bovw_scaled, train_labels_sift)
        train_pred = svm.predict(train_bovw_scaled)
        train_acc = accuracy_score(train_labels_sift, train_pred)
        val_pred = svm.predict(val_bovw_scaled)
        val_acc = accuracy_score(val_labels_sift, val_pred)
        results_bovw_svm.append([C,"rbf",gamma,train_acc,val_acc])

        if val_acc > best_acc_bovw_svm:
            best_acc_bovw_svm = val_acc
            best_C_bovw_svm = C
            best_kernel_bovw_svm = "rbf"
            best_gamma_bovw_svm = gamma
            best_model_bovw_svm = svm

best_val_pred_bovw_svm = best_model_bovw_svm.predict(val_bovw_scaled)




results_bovw_svm_df = pd.DataFrame(results_bovw_svm, 
                                  columns= ["C","Kernel","Gamma","Train Acc", "Validation Acc"])
print(results_bovw_svm_df)




conf_matrix_bovw_svm = confusion_matrix(val_labels, best_val_pred_bovw_svm)
print("Confusion matrix:")
print(conf_matrix_bovw_svm)

show_conf_matrix_bovw_svm = ConfusionMatrixDisplay(
    confusion_matrix = conf_matrix_bovw_svm,
    display_labels = best_model_bovw_svm.classes_)
show_conf_matrix_bovw_svm.plot()
plt.show()



# graph for linear kernel - hyperparameter C

results_linear = results_bovw_svm_df[results_bovw_svm_df["Kernel"] == "linear"]
plt.plot(
    results_linear["C"],
    results_linear["Validation Acc"], marker ="o"
)
plt.title("SVM + SIFT/BoVW - Linear Kernel")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Validation Accuracy")
plt.grid()
plt.show()



rbf_results = results_bovw_svm_df[results_bovw_svm_df["Kernel"] == "rbf"]
for gamma in gamma_grid:
    data = rbf_results[rbf_results["Gamma"] == gamma]
    plt.plot(data["C"], data["Validation Acc"], marker="o", label=f"gamma={gamma}")

plt.title("SVM + SIFT/BoVW - RBF Kernel")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid()
plt.show()




# test data - HOG
test_data_hog, test_labels = apply_hog_on_split("Test")
test_data_hog_scaled = scaler_hog.transform(test_data_hog)
print(test_data_hog.shape)




# test data - SIFT/BoVW
test_sift_descriptors, test_labels_sift = extract_sift_features_from_split("Test")
test_bovw = bovw_features(test_sift_descriptors)
test_bovw_scaled = scaler_bovw.transform(test_bovw)
print(test_bovw.shape)




test_pred_lr_hog = best_model.predict(test_data_hog_scaled)
test_acc_lr_hog = accuracy_score(test_labels, test_pred_lr_hog)

print(f"LR + HOG Test Accuracy: {test_acc_lr_hog}")

conf_matrix_lr_hog_test = confusion_matrix(test_labels, test_pred_lr_hog)
print(conf_matrix_lr_hog_test)

show_conf_matrix_lr_hog_test = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_lr_hog_test,
                                                      display_labels=best_model.classes_)
show_conf_matrix_lr_hog_test.plot()
plt.title("Logistic Regression + HOG - Test")
plt.show()




# test - lr + sift
test_pred_lr_bovw = best_model_bovw.predict(test_bovw_scaled)
test_acc_lr_bovw = accuracy_score(test_labels_sift, test_pred_lr_bovw)

print(f"LR + SIFT/BoVW Test Accuracy: {test_acc_lr_bovw}")

conf_matrix_lr_bovw_test = confusion_matrix(test_labels_sift, test_pred_lr_bovw)
print(conf_matrix_lr_bovw_test)

show_conf_matrix_lr_bovw_test = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_lr_bovw_test,
                                                      display_labels=best_model_bovw.classes_)
show_conf_matrix_lr_bovw_test.plot()
plt.title("Logistic Regression + SIFT/BoVW - Test")
plt.show()




# test - SVM + HOG
test_pred_svm_hog = best_model.predict(test_data_hog_scaled)
test_acc_svm_hog = accuracy_score(test_labels, test_pred_svm_hog)

print(f"SVM + HOG Test Accuracy: {test_acc_svm_hog}")

conf_matrix_svm_hog_test = confusion_matrix(test_labels, test_pred_svm_hog)
print(conf_matrix_svm_hog_test)

show_conf_matrix_svm_hog_test = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_svm_hog_test,
                                                      display_labels=best_model.classes_)
show_conf_matrix_svm_hog_test.plot()
plt.title("SVM + HOG - Test")
plt.show()




# test - svm + sift/bovw
test_pred_svm_bovw = best_model_bovw_svm.predict(test_bovw_scaled)
test_acc_svm_bovw = accuracy_score(test_labels_sift, test_pred_svm_bovw)

print(f"SVM + SIFT/BoVW Test Accuracy: {test_acc_svm_bovw}")

conf_matrix_svm_bovw_test = confusion_matrix(test_labels_sift, test_pred_svm_bovw)
print(conf_matrix_svm_bovw_test)

show_conf_matrix_svm_bovw_test = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_svm_bovw_test,
                                                      display_labels=best_model_bovw_svm.classes_)
show_conf_matrix_svm_bovw_test.plot()
plt.title("SVM + SIFT/BoVW - Test")
plt.show()
