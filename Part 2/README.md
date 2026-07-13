Note: For few coding and logics referred the offficial documenation of Pandas :
https://pandas.pydata.org/docs/


Task 1 – Feature Matrix and Target Definition

Loaded the cleaned dataset (cleaned_data.csv) from Part 1.

Features (X):
City,season,avg_temp,max_temp,min_temp,wind_speed,air_pressure,ONI
Lables(y)
AQI was selected as the regression target (y_reg).
AQI_Bucket was converted into a binary classification target (y_clf) using the following mapping:
0: Good, Satisfactory, Moderate (Low Risk)
1: Poor, Very Poor, Severe (High Risk)


Task 2 – Encoding Categorical Features

 ENSO_Category feature from the ONI values using the NOAA Oceanic Niño Index (ONI) classificat done One-Hot Encoding to City, season, and ENSO_Category as these categorical features have no natural order.One-Hot Encoding prevents the model from assuming a false ordinal relationship between categorical values like for example 
Label encoding turns categories into numbers like:
 ordered it assumesone is  "greater than other and twice as far from Winter (0) as Summer is. But seasons have no real rank or distance like that
One-hot encoding fixes this by giving each category its own 0/1 column.

Task 3 – Train-Test Split and Feature Scaling

Splited the dataset into 80% training and 20% testing using train_test_split(test_size=0.2, random_state=42).
Applied StandardScaler by fitting it only on the training data and then transforming both the training and test sets.
The scaler was not fitted on the entire dataset to prevent data leakage. Fitting on the full dataset would allow the training process to use the test set's mean and standard deviation, introducing information from unseen data and producing overly optimistic evaluation results.
Since we use 2 model as a ML practice purpose used 2 scalers for each separately.


Task 4 – Linear Regression and Ridge Regression

Trained a Linear Regression model using the scaled training data and predicted AQI values for the test dataset.Evaluated the model using Mean Squared Error (MSE) and R² Score.
MSE = 9308.28 indicates the average squared difference means a few large outlier errors are resulting in this value
R² = 0.61 indicates that the model explains approximately 61% of the variation in AQI using the selected features.
printed the regression coefficients with their corresponding feature names and identified the three features with the largest absolute coefficient values.
positive coefficient =  increasing the feature increases the predicted AQI.
negative coefficient = increasing the feature decreases the predicted AQI.
Feature City_Amritsar Coefficient -101.52 Feature City_Bengaluru Coefficient -101.04 FeatureCity_Hyderabad Coefficient-100.15
Trained a Ridge Regression model using alpha = 1.0 and evaluated it using the same metrics.

Ridge Regression produced almost the same performance as Linear Regression, indicating that regularization had minimal impact on this dataset.
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_records.html#pandas.DataFrame.from_records
Ridge applies L2 regularization, which reduces the magnitude of coefficients to improve model stability and reduce overfitting. The alpha parameter controls the strength of regularization; higher values apply stronger coefficient shrinkage
Model      MSE  R2 score
0  Linear Regression  9308.28      0.61
1   Ridge Regression   9309.11      0.61


Task 5 – Logistic Regression

AQI_Bucket
1    79.24
0    20.76 Based on the value counts one  class having fewer than 35% of samples so will use **class_weight="balance"** beacause SMOTE creates synthetic points before your train/test split (if done wrong), which can leak information into your test set and inflate your accuracy artificially. class_weight never touches your data, so this risk doesn't exist.
Since class_weight reweights the loss function rather than resampling the data, class counts remain unchanged — still 79.24% / 20.76% after training. What changes internally is that each High-Risk sample contributes ~3.8x more to the model's error calculation, forcing it to pay attention to the minority class without altering the dataset itself.
Logistic Regression model (max_iter=1000) was trained using the scaled training data. The model achieved an accuracy of 88.86% and an AUC of 0.9504, showing that it can distinguish between Low-Risk and High-Risk AQI categories effectively. Precision, Recall, F1-score, the Confusion Matrix, and the ROC curve were used to evaluate the model.



 Confusion Matrix:
                   Pred: Low risk  Pred: High risk
Actual: Low risk             1871              230
Actual: High risk              63              467

 I labeled Low rosk as 0 and High risk as 1  on the binary classifcation accoridng to CM r
TP = 467: High-Risk AQI days correctly predicted as High Risk.
FN = 63: High-Risk AQI days incorrectly predicted as Low Risk. This is the most critical error because unhealthy air quality is classified as safe. 
FP = 230: Low-Risk AQI days incorrectly predicted as High Risk, resulting in an unnecessary warning.
TN = 1871: Low-Risk AQI days correctly predicted as Low Risk.

Precision = TP / (TP + FP)

Precision measures how many predicted High-Risk days were actually High Risk. The model achieved a precision of 0.67, meaning 67% of the High-Risk predictions were correct.

Recall = TP / (TP + FN)

Recall measures how many actual High-Risk days were correctly identified. The model achieved a recall of 0.88, meaning it successfully detected 88% of all High-Risk AQI days.

for my task, Recall is more important than Precision because missing a High-Risk AQI day (False Negative) is more harmful than issuing a false warning. A missed High-Risk day may lead people to believe the air quality is safe when it is actually unhealthy.

The model achieved an AUC of 0.9504, indicating an excellent ability to distinguish between Low-Risk and High-Risk AQI categories. An AUC value closer to 1.0 indicates better class separation, while 0.5 represents random guessing.

Task 5(b) – Decision Threshold Sensitivity

The Logistic Regression model predicts probabilities for the positive class (High Risk). By default, predictions are classified using a threshold of 0.50. To study the effect of changing the decision boundary, thresholds from 0.30 to 0.70 were evaluated, 
threshold with the highest F1-score was selected as the best balance between Precision and Recall for this dataset.

To improve Recall, the decision threshold can be lowered, allowing the model to identify more High-Risk days this also increases the number of False Positives, resulting in more unnecessary High-Risk warnings.

Task 6-Regularization Experiment on Logistic Regression

A second Logistic Regression model was trained using C = 0.01, which applies stronger L2 regularization, and its performance was compared with the baseline model (C = 1.0).

Model	Precision	Recall	AUC
Logistic Regression (C = 1.0)	0.67	0.881	0.950
Logistic Regression (C = 0.01)	0.63	0.874	0.943

The C parameter controls the strength of regularization in Logistic Regression. A smaller C applies stronger regularization, shrinking the model coefficients more to reduce overfitting, while a larger C applies weaker regularization.

For this dataset, reducing C from 1.0 to 0.01 slightly worsened the model's performance, as Precision, Recall, and AUC all decreased. Therefore, the baseline model (C = 1.0) performed better for predicting AQI risk categories.



Task 7 Bootstrap Confidence Interval for AUC Difference

bootstrap analysis with 500 resamples was performed to compare the AUC of the baseline Logistic Regression model (C = 1.0) with the strongly regularized model (C = 0.01). For each bootstrap sample, the difference in AUC (AUC(C = 1.0) − AUC(C = 0.01)) was calculated.

Mean AUC Difference: 0.0078
95% Confidence Interval: [0.0046, 0.0107]

Since the 95% confidence interval does not include zero, the performance advantage of the C = 1.0 model is likely consistent across different data samples. Therefore, the baseline model provides a more reliable classification performance than the model with C = 0.01.
