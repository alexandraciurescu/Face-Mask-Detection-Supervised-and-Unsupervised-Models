

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

# k-means + hog features

from sklearn.preprocessing import StandardScaler
from sklearn. cluster import KMeans
from sklearn.metrics import adjusted_rand_score

scaler_hog = StandardScaler()
train_hog_scaled = scaler_hog.fit_transform(train_data_hog)
val_hog_scaled = scaler_hog.transform(val_data_hog)

k_grid = [2,3,4]
init_grid = ["k-means++", "random"]
n_init_grid = [5,10,20]

results_hog_kmeans = []
best_ari_hog_kmeans = -1
best_init_hog_kmeans = None
best_model_hog_kmeans = None
best_n_init_hog_kmeans = None

for k in k_grid:
    for init in init_grid:
        for n_init in n_init_grid:
            kmeans = KMeans(n_clusters=k, init=init, n_init=n_init, random_state=42, max_iter=1000)
            kmeans.fit(train_hog_scaled)
            train_clusters = kmeans.labels_
            val_clusters = kmeans.predict(val_hog_scaled)

            train_ari = adjusted_rand_score(train_labels,train_clusters)
            val_ari = adjusted_rand_score(val_labels, val_clusters)

            results_hog_kmeans.append([k,init,n_init,train_ari,val_ari])

            if val_ari > best_ari_hog_kmeans:
                best_ari_hog_kmeans = val_ari
                best_k_hog_kmeans = k
                best_init_hog_kmeans = init
                best_model_hog_kmeans = kmeans
                best_n_init_hog_kmeans = n_init

print(f"Best k: {best_k_hog_kmeans}")
print(f"Best init: {best_init_hog_kmeans}")
print(f"Best n_init: {best_n_init_hog_kmeans}")
print(f"Best validation ARI: {best_ari_hog_kmeans}")


import pandas as pd
results_hog_kmeans_df = pd.DataFrame(results_hog_kmeans, columns=["K", "Init",
                                                                  "N Init", "Train ARI", "Validation ARI"])
print(results_hog_kmeans_df)

# graphs
# fix n_init = 10

results_n10 = results_hog_kmeans_df[results_hog_kmeans_df["N Init"] == 10]


for init in init_grid:
        data = results_n10[results_n10["Init"] == init]
            
        plt.plot( data["K"], data["Validation ARI"],
                 marker="o", label = init)
        
plt.xlabel("Number of clusters")
plt.ylabel("Validation ARI")
plt.title("K-Means + HOG - Effect of K and Initialization")
plt.legend()
plt.grid()
plt.show()

# fix k=2

results_k2 = results_hog_kmeans_df[results_hog_kmeans_df["K"] == 2]
for init in init_grid:
    data = results_k2[results_k2["Init"]==init]
    plt.plot(data["N Init"], data["Validation ARI"], marker="o", label=init)

plt.xlabel("n_init")
plt.ylabel("Validation ARI")
plt.title("K-Means + HOG - Effect of n_init")
plt.legend()
plt.grid()
plt.show()

best_val_clusters_hog_kmeans = best_model_hog_kmeans.predict(val_hog_scaled)
cluster_table_hog = pd.crosstab(val_labels, best_val_clusters_hog_kmeans, rownames=["Real class"], colnames=["Cluster"])
print(cluster_table_hog)

# k-means + sift/bovw

scaler_bovw = StandardScaler()
train_bovw_scaled = scaler_bovw.fit_transform(train_bovw)
val_bovw_scaled = scaler_bovw.transform(val_bovw)

k_grid = [2,3,4]
init_grid = ["k-means++", "random"]
n_init_grid = [5,10,20]

results_bovw_kmeans = []
best_ari_bovw_kmeans = -1
best_init_bovw_kmeans = None
best_model_bovw_kmeans = None
best_n_init_bovw_kmeans = None
best_k_bovw_kmeans = None

for k in k_grid:
    for init in init_grid:
        for n_init in n_init_grid:
            kmeans = KMeans(n_clusters=k, init=init, n_init=n_init, random_state=42, max_iter=1000)

            kmeans.fit(train_bovw_scaled)

            train_clusters = kmeans.labels_
            val_clusters = kmeans.predict(val_bovw_scaled)

            train_ari = adjusted_rand_score(train_labels_sift,train_clusters)
            val_ari = adjusted_rand_score(val_labels_sift, val_clusters)

            results_bovw_kmeans.append([k,init,n_init,train_ari,val_ari])

            if val_ari > best_ari_bovw_kmeans:
                best_ari_bovw_kmeans = val_ari
                best_k_bovw_kmeans = k
                best_init_bovw_kmeans = init
                best_model_bovw_kmeans = kmeans
                best_n_init_bovw_kmeans = n_init

print(f"Best k: {best_k_bovw_kmeans}")
print(f"Best init: {best_init_bovw_kmeans}")
print(f"Best n_init: {best_n_init_bovw_kmeans}")
print(f"Best validation ARI: {best_ari_bovw_kmeans}")


results_bovw_kmeans_df = pd.DataFrame(results_bovw_kmeans, columns=["K", "Init",
                                                                  "N Init", "Train ARI", "Validation ARI"])
print(results_bovw_kmeans_df)

# graphs
# fix n_init = 10

results_n10 = results_bovw_kmeans_df[results_hog_kmeans_df["N Init"] == 10]


for init in init_grid:
        data = results_n10[results_n10["Init"] == init]
            
        plt.plot( data["K"], data["Validation ARI"],
                 marker="o", label = init)
        
plt.xlabel("Number of clusters")
plt.ylabel("Validation ARI")
plt.title("K-Means + SIFT/BoVW - Effect of K and Initialization")
plt.legend()
plt.grid()
plt.show()

best_val_clusters_bovw_kmeans = best_model_bovw_kmeans.predict(val_bovw_scaled)
cluster_table_bovw = pd.crosstab(val_labels, best_val_clusters_bovw_kmeans, rownames=["Real class"], colnames=["Cluster"])
print(cluster_table_bovw)

# hierarchical clustering + hog

n_clusters_grid = [2,3,4]
linkage_grid= ["single", "complete", "average", "ward"]
metric_grid = ["euclidean", "manhattan", "cosine"]

from sklearn.cluster import AgglomerativeClustering

results_hog_hierarchical = []
best_ari_hog_hierarchical = -1
best_linkage_hog_hierarchical = None
best_metric_hog_hierarchical = None
best_k_hog_hierarchical = None

for k in n_clusters_grid:
    for linkage in linkage_grid:
        for metric in metric_grid:
            if linkage=="ward" and metric!="euclidean": continue

            model = AgglomerativeClustering(n_clusters=k,
                                            linkage=linkage,
                                            metric=metric)
            val_clusters = model.fit_predict(val_hog_scaled)
            val_ari = adjusted_rand_score(val_labels,val_clusters)
            results_hog_hierarchical.append([k, linkage,metric, val_ari])

            if val_ari>best_ari_hog_hierarchical:
                best_ari_hog_hierarchical = val_ari
                best_k_hog_hierarchical = k
                best_linkage_hog_hierarchical = linkage
                best_metric_hog_hierarchical = metric

print(f"Best k: {best_k_hog_hierarchical}")
print(f"Best linkage: {best_linkage_hog_hierarchical}")
print(f"Best mestric: {best_metric_hog_hierarchical}")
print(f"Best validation ARI: {best_ari_hog_hierarchical}")


results_hog_hierarchical_df = pd.DataFrame(results_hog_hierarchical, 
                                           columns=["K", "Linkage", "Metric", "Validation ARI"])
print(results_hog_hierarchical_df)

for linkage in linkage_grid:
    for metric in metric_grid:
        if linkage=="ward" and metric!="euclidean":continue
        data = results_hog_hierarchical_df[(results_hog_hierarchical_df["Linkage"] == linkage) &
                                           (results_hog_hierarchical_df["Metric"] == metric)]
        plt.plot(data["K"], data["Validation ARI"], marker="o",label=f"{linkage} +{metric}")

plt.xlabel("Number of clusters")
plt.ylabel("Validation ARI")
plt.title("Hierarchical Clustering + HOG - Hyperparameter Tuning")
plt.legend()
plt.grid()
plt.show()

best_hierarchical_hog = AgglomerativeClustering(
    n_clusters=best_k_hog_hierarchical,
    linkage=best_linkage_hog_hierarchical,
    metric=best_metric_hog_hierarchical
)
best_val_clusters_hog_hierarchical = best_hierarchical_hog.fit_predict(val_hog_scaled)
cluster_table_hog_hierarchical = pd.crosstab(val_labels, best_val_clusters_hog_hierarchical,
                                             rownames=["Real class"], colnames=["Cluster"])
print(cluster_table_hog_hierarchical)

# hierarchical clustering + SIFT/BoVW

n_clusters_grid = [2,3,4]
linkage_grid = ["single", "complete", "average", "ward"]
metric_grid = ["euclidean", "manhattan", "cosine"]

results_bovw_hierarchical = []
best_ari_bovw_hierarchical = -1
best_linkage_bovw_hierarchical = None
best_metric_bovw_hierarchical = None
best_k_bovw_hierarchical = None

for k in n_clusters_grid:
    for linkage in linkage_grid:
        for metric in metric_grid:
            if linkage=="ward" and metric!="euclidean": continue

            model = AgglomerativeClustering(n_clusters=k,
                                            linkage=linkage,
                                            metric=metric)
            val_clusters = model.fit_predict(val_bovw_scaled)
            val_ari = adjusted_rand_score(val_labels_sift,val_clusters)
            results_bovw_hierarchical.append([k, linkage,metric, val_ari])

            if val_ari>best_ari_bovw_hierarchical:
                best_ari_bovw_hierarchical = val_ari
                best_k_bovw_hierarchical = k
                best_linkage_bovw_hierarchical = linkage
                best_metric_bovw_hierarchical = metric

print(f"Best k: {best_k_bovw_hierarchical}")
print(f"Best linkage: {best_linkage_bovw_hierarchical}")
print(f"Best mestric: {best_metric_bovw_hierarchical}")
print(f"Best validation ARI: {best_ari_bovw_hierarchical}")

results_bovw_hierarchical_df = pd.DataFrame(results_bovw_hierarchical, 
                                           columns=["K", "Linkage", "Metric", "Validation ARI"])
print(results_bovw_hierarchical_df)

for linkage in linkage_grid:
    for metric in metric_grid:
        if linkage=="ward" and metric!="euclidean":continue
        data = results_bovw_hierarchical_df[(results_bovw_hierarchical_df["Linkage"] == linkage) &
                                           (results_bovw_hierarchical_df["Metric"] == metric)]
        plt.plot(data["K"], data["Validation ARI"], marker="o",label=f"{linkage} +{metric}")

plt.xlabel("Number of clusters")
plt.ylabel("Validation ARI")
plt.title("Hierarchical Clustering + SIFT/BoVW - Hyperparameter Tuning")
plt.legend()
plt.grid()
plt.show()

best_hierarchical_bovw = AgglomerativeClustering(
    n_clusters=best_k_bovw_hierarchical,
    linkage=best_linkage_bovw_hierarchical,
    metric=best_metric_bovw_hierarchical
)
best_val_clusters_bovw_hierarchical = best_hierarchical_bovw.fit_predict(val_bovw_scaled)
cluster_table_bovw_hierarchical = pd.crosstab(val_labels, best_val_clusters_bovw_hierarchical,
                                             rownames=["Real class"], colnames=["Cluster"])
print(cluster_table_bovw_hierarchical)

# random chance
np.random.seed(42)
random_results=[]
for k in [2,4]:
    ari_scores=[]
    for i in range(100):
        random_clusters = np.random.randint(0,k,len(val_labels))
        ari = adjusted_rand_score(val_labels, random_clusters)
        ari_scores.append(ari)

    print(f"K={k}, Random ARI: {np.mean(ari_scores)}")  


# extracting features for test data

test_data_hog, test_labels = apply_hog_on_split("Test")


test_sift_descriptors, test_labels_sift = extract_sift_features_from_split("Test")
test_bovw = bovw_features(test_sift_descriptors)


test_hog_scaled = scaler_hog.transform(test_data_hog)
test_bovw_scaled = scaler_bovw.transform(test_bovw)

test_clusters_hog_kmeans = best_model_hog_kmeans.predict(test_hog_scaled)
test_ari_hog_kmeans = adjusted_rand_score(test_labels, test_clusters_hog_kmeans)
print(f"K-means + HOG Test ARI: {test_ari_hog_kmeans}")

cluster_table_test_hog_kmeans = pd.crosstab(test_labels,
                                            test_clusters_hog_kmeans,
                                            rownames=["Real class"],
                                            colnames=["Cluster"]
                                            )
print(cluster_table_test_hog_kmeans)

test_clusters_bovw_kmeans = best_model_bovw_kmeans.predict(test_bovw_scaled)
test_ari_bovw_kmeans = adjusted_rand_score(test_labels_sift, test_clusters_bovw_kmeans)

print(f"K-Means + SIFT/BoVW Test ARI: {test_ari_bovw_kmeans}")

cluster_table_test_bovw_kmeans = pd.crosstab(test_labels_sift,
                                            test_clusters_bovw_kmeans,
                                            rownames=["Real class"],
                                            colnames=["Cluster"]
                                            )
print(cluster_table_test_bovw_kmeans)

test_hierarchical_hog = AgglomerativeClustering(
    n_clusters=best_k_hog_hierarchical,
    linkage=best_linkage_hog_hierarchical,
    metric = best_metric_hog_hierarchical
)
test_clusters_hog_hierarchical = test_hierarchical_hog.fit_predict(test_hog_scaled)
test_ari_hog_hierarchical = adjusted_rand_score(test_labels, test_clusters_hog_hierarchical)

print(f"Hierarchical + HOG Test ARI: {test_ari_hog_hierarchical}")

cluster_table_test_hog_hierarchical = pd.crosstab(test_labels,
                                                  test_clusters_hog_hierarchical,
                                                  rownames=["Real class"], colnames=["Cluster"])
print(cluster_table_test_hog_hierarchical)

test_hierarchical_bovw = AgglomerativeClustering(
    n_clusters=best_k_bovw_hierarchical,
    linkage=best_linkage_bovw_hierarchical,
    metric = best_metric_bovw_hierarchical
)
test_clusters_bovw_hierarchical = test_hierarchical_bovw.fit_predict(test_bovw_scaled)
test_ari_bovw_hierarchical = adjusted_rand_score(test_labels_sift, test_clusters_bovw_hierarchical)

print(f"Hierarchical + SIFT/BoVW Test ARI: {test_ari_bovw_hierarchical}")

cluster_table_test_bovw_hierarchical = pd.crosstab(test_labels_sift,
                                                  test_clusters_bovw_hierarchical,
                                                  rownames=["Real class"], colnames=["Cluster"])
print(cluster_table_test_bovw_hierarchical)

# analyze clusters

test_paths =[]
for class_name in ["WithMask","WithoutMask"]:
    class_path = data_path / "Test" / class_name
    for img_path in class_path.iterdir():
        test_paths.append(img_path)

def show_cluster_images(cluster_labels,title,majority_or_minority_image=False):
    clusters = np.unique(cluster_labels)
    plt.figure(figsize=(6,3*len(clusters)))
    plot_nr=1
    for c in clusters:
        indexes = np.where(cluster_labels==c)[0]
        if majority_or_minority_image:
            labels_cluster = test_labels[indexes]
            labels,counts=np.unique(labels_cluster,return_counts=True)

            majority_label=labels[np.argmax(counts)]
            if majority_label == "WithMask": minority_label="WithoutMask"
            else: minority_label="WithMask"
            majority_index=indexes[np.where(test_labels[indexes]==majority_label)[0][0]]
            minority_index=indexes[np.where(test_labels[indexes]==minority_label)[0][0]]

            selected = [(majority_index, "Majority"),
                        (minority_index, "Minority")]

        else:  selected = [(indexes[0], "Majority"),
                            (indexes[1], "Minority")]

        for index, example_type in selected:
            plt.subplot(len(clusters),2,plot_nr)
            img=Image.open(test_paths[index])
            plt.imshow(img)
            if majority_or_minority_image:
                plt.title(f"Cluster {c} - {example_type}\n"
                        f"{test_labels[index]}")
            else: plt.title(f"Cluster {c}\n{test_labels[index]}")
            plt.axis("off")
            plot_nr+=1

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


show_cluster_images(test_clusters_hog_kmeans, "K-Means + HOG", True)

show_cluster_images(test_clusters_hog_hierarchical, "Hierarchical + HOG", True)

show_cluster_images(test_clusters_bovw_kmeans, "K-Means + SIFT/BoVW")

show_cluster_images(test_clusters_bovw_hierarchical, "Hierarchical + SIFT/BoVW")
