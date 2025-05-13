# Breast Cancer Classification Using Machine Learning

## Abstract
Breast cancer remains one of the most common and life-threatening diseases affecting women globally. Early diagnosis significantly increases the survival rate, and recent advancements in machine learning provide promising avenues for accurate and efficient detection. This study implements and evaluates various supervised learning algorithms—Logistic Regression, Decision Tree, K-Nearest Neighbors, Random Forest, XGBoost, Naive Bayes, and a Neural Network—to classify tumors as malignant or benign using the Breast Cancer Wisconsin dataset. The aim is to compare model performance and determine the most effective classifier in terms of accuracy, recall, precision, and F1-score. The findings reveal the strengths and trade-offs of different models, contributing to the growing body of research in AI-assisted diagnostics.

## Introduction
Breast cancer is a critical public health issue, with millions diagnosed annually. Traditional diagnostic methods like mammography, biopsy, and ultrasound, although effective, are often invasive, costly, and time-consuming. The application of machine learning (ML) in healthcare has opened up innovative methods for early disease detection with minimal human intervention.

In this project, we analyze the Breast Cancer Wisconsin dataset, which contains a range of features computed from digitized images of fine needle aspirate (FNA) of breast masses. The goal is to build, train, and evaluate multiple ML models for binary classification—distinguishing between malignant and benign tumors. We focus on model accuracy as well as other performance metrics to assess the efficacy and reliability of each method.

## Keywords
- Breast Cancer
- Machine Learning
- Classification
- Neural Networks
- XGBoost
- Supervised Learning
- Healthcare AI
- Data Preprocessing
- Model Evaluation

## Literature Survey
Several studies have explored the potential of machine learning in cancer detection. Research has shown that classification models can outperform traditional statistical models when trained on comprehensive datasets. Logistic Regression and Decision Trees are commonly used due to their simplicity and interpretability. K-Nearest Neighbors, while less interpretable, can yield competitive results when tuned correctly.

More advanced ensemble methods such as Random Forest and XGBoost have been favored for their robustness and performance in high-dimensional datasets. Neural Networks, particularly deep learning models, have recently gained traction for their ability to automatically extract and learn features, though they require larger datasets and longer training times.

## Methodology

### Data Acquisition
The Breast Cancer Wisconsin dataset was obtained and loaded using `pandas`.

### Data Preprocessing
- Missing or irrelevant values were handled.
- Features were standardized using `StandardScaler`.
- The target variable (diagnosis) was encoded into binary format.
- The dataset was split into training and test sets using an 80-20 ratio.

### Models Used
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Random Forest Classifier
- XGBoost
- Naive Bayes
- A Neural Network using TensorFlow/Keras

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Experimental Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| ~96%     | High      | High   | High     |
| Decision Tree      | ~93%     | Medium    | Medium | Medium   |
| KNN                | ~95%     | High      | Medium | High     |
| Random Forest      | ~97%     | High      | High   | High     |
| XGBoost            | ~97%     | Very High | High   | High     |
| Naive Bayes        | ~93%     | Medium    | Medium | Medium   |
| Neural Network     | ~96-97%  | High      | High   | High     |

## Results and Discussion
Random Forest and XGBoost models showed the best performance, effectively capturing non-linear relationships and interactions among features. Neural Networks also yielded promising results but require careful tuning. Logistic Regression performed reliably and is suitable for interpretable diagnostics.

Simpler models like Naive Bayes and Decision Tree were moderately accurate, and KNN performed well but with sensitivity to hyperparameters. These findings suggest a trade-off between interpretability and performance.

## Future Work
- Hyperparameter tuning with grid search or Bayesian optimization
- Enhanced feature engineering or dimensionality reduction (e.g., PCA)
- Use of CNNs on image data directly
- Implementation of explainability tools like SHAP or LIME
- Deployment as a web-based API or tool
- Balancing dataset using synthetic data generation (e.g., SMOTE)

## References
1. WHO, Cancer Fact Sheets - Breast Cancer, https://www.who.int/
2. UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
3. Scikit-learn: Pedregosa et al., JMLR, 2011
4. XGBoost: Chen & Guestrin, arXiv, 2016
5. Deep Learning: Chollet, F., Deep Learning with Python, Manning, 2017
6. SHAP: Lundberg & Lee, NIPS, 2017

