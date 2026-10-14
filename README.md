# 🌌 Pulsar Star Detection with Support Vector Machines

Classifying real pulsar signals from noise in radio-telescope candidate data with an SVM, with a comparison of kernels, hyperparameter tuning and an evaluation that accounts for heavy class imbalance.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

## Problem
Pulsars are rare neutron stars that emit radio waves. Most detections are noise or radio-frequency interference, so each candidate must be classified as a real pulsar or not. In this dataset only **1,639 of 17,898 candidates (9.2%)** are real pulsars, so a model that always says "not a pulsar" is already 90.8% accurate. Accuracy alone is therefore not enough to judge a model.

## Dataset
- 17,898 candidates, 8 continuous features and a binary target (`target_class`: 0 = noise, 1 = pulsar)
- Features: mean, standard deviation, excess kurtosis and skewness of the integrated pulse profile (IP) and of the DM-SNR curve
- No missing values; all features are skewed and contain outliers

## Method
1. Clean and rename columns, check class balance (90.8% / 9.2%)
2. EDA with boxplots and histograms
3. 80/20 train–test split, `StandardScaler` fitted on the training set only
4. SVM with RBF, linear, polynomial and sigmoid kernels at several values of `C`
5. Compared accuracy with the **null accuracy** (0.9235) and checked train vs test for overfitting
6. Confusion matrix, classification report, ROC AUC and cross-validation
7. `GridSearchCV` over kernel, `C`, `gamma` and `degree`

## Results

| Model | Test accuracy |
|---|---|
| RBF, C = 1 (default) | 0.9827 |
| RBF, C = 100 | 0.9832 |
| RBF, C = 1000 | 0.9816 |
| Linear, C = 1 | 0.9830 |
| Linear, C = 100 / 1000 | 0.9832 |
| Polynomial, C = 1 / 100 | 0.9807 / 0.9824 |
| Sigmoid, C = 1 / 100 | 0.8858 / 0.8855 |
| **GridSearchCV best (RBF, C = 10, gamma = 0.3)** | **0.9835** (5-fold CV 0.9793) |

- Null accuracy: **0.9235**, so the model is a real improvement over always predicting "no pulsar"
- Linear SVM: train 0.9783 vs test 0.9830, so no sign of overfitting

**Pulsar class (class 1) on the test set** (linear SVM, C = 1):

| Precision | Recall | F1 | Support |
|---|---|---|---|
| 0.93 | 0.84 | 0.88 | 274 |

Confusion matrix (rows = actual, columns = predicted): `[[3289, 17], [44, 230]]`: 17 false alarms and 44 missed pulsars out of 274.

## Limitations and next steps
- Recall for pulsars is 0.84; try `class_weight="balanced"` and compare precision vs recall.
- Add a precision–recall curve and PR-AUC, which are better than ROC for imbalanced data.
- Use `StratifiedKFold` with a `Pipeline(StandardScaler, SVC)` for cross-validation.
- Compare with tree-based models such as Random Forest or Gradient Boosting.

## Run it
```bash
git clone https://github.com/vishnu-chaithanya-ds/pulsar-star-detection-svm.git
cd pulsar-star-detection-svm
pip install -r requirements.txt
# add data/pulsar_stars.csv (see data/README.md)
jupyter notebook notebooks/pulsar_star_detection_svm.ipynb
```

## Author
**Tadipatri Vishnu Chaithanya** · [Portfolio](https://vishnuchaithanya.netlify.app/) · [LinkedIn](https://www.linkedin.com/in/tadipatri-vishnu-chaithanya-52817037b) · [GitHub](https://github.com/vishnu-chaithanya-ds)
