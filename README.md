# Dry Bean Classification: ML Assignment 2

**Live app:** https://7aterkjvzwz3ayzvphecpv.streamlit.app/
**GitHub repository:** https://github.com/HarshNagarkar17/ml

## a. Problem statement

Dry beans of different varieties look alike to the eye, and sorting them by hand is slow and inconsistent. The goal is to classify a dry bean grain into one of **7 varieties** using **16 shape and size measurements** extracted from images of the grain. This is a multi-class classification problem. Five classifiers are trained on the same data, evaluated with six metrics, and compared. The trained models are served through a Streamlit app where a test CSV can be uploaded and evaluated without retraining.

## b. Dataset description

- **Source:** Dry Bean Dataset (Koklu & Ozkan, 2020), downloaded from Kaggle. The same data is hosted on the UCI Machine Learning Repository.
- **Instances:** 13,611 raw rows. After dropping 68 duplicate rows, 13,543 rows were used.
- **Features:** 16 numeric features (minimum required: 12). These are Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3 and ShapeFactor4. The `AspectRation` spelling comes from the original dataset.
- **Target:** `Class`, with 7 classes: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA.
- **Class balance (raw data):** DERMASON 3546, SIRA 2636, SEKER 2027, HOROZ 1928, CALI 1630, BARBUNYA 1322, BOMBAY 522. The classes are imbalanced, with BOMBAY the smallest.
- **Missing values:** none.
- **Split:** 80/20 stratified split with `random_state=42`, giving 10,834 training rows and 2,709 test rows. The test split is saved as `test_data.csv`.
- **Preprocessing:** `StandardScaler` is part of the saved pipeline for Logistic Regression, KNN and SVM. The Decision Tree and Naive Bayes use the raw features. The target is label-encoded, and the encoder is saved in `model/artifacts.joblib`.

## c. GitHub repository

https://github.com/HarshNagarkar17/ml

```
├── app.py                  # Streamlit app (loads saved models, no retraining)
├── train_model.py          # trains and saves all five models, metrics and test split
├── requirements.txt
├── README.md
├── test_data.csv           # 2,709-row held-out test set
├── data/dry_bean.csv       # dataset from Kaggle
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── artifacts.joblib    # label encoder, feature list, model registry
    └── metrics.csv
```

## d. Models used

All five models were trained on the same training split and evaluated on the same 2,709-row test set. Precision, Recall and F1 are **weighted averages** across the 7 classes. AUC is **one-vs-rest, macro-averaged** and computed from predicted probabilities.

Hyperparameters: Logistic Regression (`max_iter=2000`), Decision Tree (`max_depth=12`), KNN (`n_neighbors=7`), Gaussian Naive Bayes (defaults), SVM (RBF kernel, `C=5`, `probability=True`).

### Comparison table

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ---------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression    | 0.9195   | 0.9947 | 0.9201    | 0.9195 | 0.9197 | 0.9028 |
| Decision Tree          | 0.9011   | 0.9621 | 0.9013    | 0.9011 | 0.9012 | 0.8804 |
| KNN                    | 0.9118   | 0.9852 | 0.9123    | 0.9118 | 0.9119 | 0.8933 |
| Naive Bayes            | 0.7630   | 0.9670 | 0.7647    | 0.7630 | 0.7607 | 0.7143 |
| Support Vector Machine | 0.9236   | 0.9946 | 0.9237    | 0.9236 | 0.9236 | 0.9076 |

### Observations

| ML Model Name                       | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression                 | Reached 0.9195 accuracy and the highest AUC (0.9947), only slightly ahead of SVM. A linear model scoring this well after scaling suggests the classes are close to linearly separable in this feature space. Its most frequent mistakes were DERMASON and SIRA being swapped (53 and 41 samples).                                                                                                                                        |
| Decision Tree                       | Accuracy of 0.9011, the lowest of the four stronger models, and a clearly lower AUC (0.9621). A tree's probabilities come from the class proportions in each leaf, so they are coarse and give a weaker ranking of classes than the smoother probabilities of the other models. It had the most DERMASON/SIRA errors (63 and 57) and the weakest SIRA F1 (0.834).                                                                        |
| KNN                                 | Scored 0.9118 accuracy and 0.9852 AUC with k=7 on scaled features. It performs between Logistic Regression and the Decision Tree. KNN depends on distances, so the scaling step matters, and its errors were again mostly DERMASON/SIRA (55 and 53) plus some BARBUNYA predicted as CALI (22).                                                                                                                                           |
| Naive Bayes                         | Clearly the weakest, at 0.7630 accuracy and 0.7143 MCC. Many features are strongly correlated (Area, Perimeter, ConvexArea, EquivDiameter and the axis lengths), which breaks the independence assumption that Naive Bayes relies on. Its AUC of 0.9670 is still decent, so it ranks classes reasonably but its hard predictions are poor, with BARBUNYA often predicted as CALI (108 samples) and BARBUNYA getting an F1 of only 0.548. |
| Support Vector Machine              | Best accuracy (0.9236), F1 (0.9236) and MCC (0.9076), with an AUC (0.9946) essentially tied with Logistic Regression. The RBF kernel on scaled features captures non-linear boundaries slightly better than the linear model. It also had the fewest DERMASON/SIRA errors among the stronger models (52 and 44).                                                                                                                         |
| **Overall winner for this dataset** | **Support Vector Machine.** It leads on accuracy, precision, recall, F1 and MCC, and ties for best AUC. The margin over Logistic Regression is small (about 0.4 percentage points of accuracy). All models classify BOMBAY perfectly (F1 = 1.0), and the main remaining difficulty for every model except Naive Bayes is separating DERMASON from SIRA.                                                                                  |

## Streamlit app

The app (`app.py`) provides:

- CSV upload (the test set; it must contain all 16 feature columns and the `Class` column).
- A model selection dropdown for the five classifiers.
- The six evaluation metrics for the selected model.
- A confusion matrix and a classification report.
- A table comparing all five models on the uploaded data.

Models are loaded from `model/` and nothing is retrained on the uploaded data.

## Run locally

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python train_model.py      # optional: retrains and rewrites model/ and test_data.csv
streamlit run app.py
```

The app is deployed on Streamlit Community Cloud with Python 3.12 and `scikit-learn==1.5.2`, the same version the models were trained with.
