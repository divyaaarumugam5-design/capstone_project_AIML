Note: SYntax and steps are refereed from offcial documenation 
https://scikit-learn.org/stable/user_guide.html

Task 1- Decision Tree
The baseline Decision Tree achieved a training accuracy of 99.99% and a test accuracy of 89.02%. The much higher training accuracy compared to the test accuracy indicates that the model is overfitting, as it has learned the training data very well but does not generalize as effectively to unseen data. Decision trees are considered high-variance models because they make greedy decisions by selecting the best split at each node without revisiting earlier splits. This allows them to fit the training data closely, but also makes them sensitive to small changes in the data, increasing the risk of overfitting.

Task 2 Controlled Decision Tree

Model	Train Accuracy	Test Accuracy
Baseline Tree	99.99%	89.02%
Controlled Tree	87.43%	87.72%

The controlled Decision Tree (max_depth=5, min_samples_split=20) achieved a training accuracy of 87.43% and a test accuracy of 87.72%. The max_depth parameter limits how deep the tree can grow, reducing variance at the cost of a small increase in bias. The min_samples_split parameter prevents a node from splitting unless it has at least 20 samples, reducing splits caused by noise in small subsets. Compared to the unconstrained tree, the controlled tree has a much smaller train/test accuracy gap, indicating better generalization and reduced overfitting.

Task 3- Gini vs Entropy 
The Decision Tree using Gini achieved a test accuracy of 87.72%, while the Entropy criterion achieved 86.05%. Gini impurity is calculated as 1 − Σ pi², and Entropy is calculated as −Σ pi log₂(pi). A Gini value of 0 means that all samples in a node belong to a single class, indicating a perfectly pure node. In this dataset, the Gini criterion performed slightly better than Entropy.

Task 4- Ensembling model 
The Random Forest model achieved a training accuracy of 91.87%, a test accuracy of 89.43%, and a ROC-AUC of 0.9408. The top five most important features were min_temp, wind_speed, max_temp, avg_temp, and air_pressure. Random Forest computes feature importance by measuring the average reduction in Gini impurity contributed by each feature across all splits in all trees. This differs from a linear regression coefficient, which represents the direction and magnitude of a feature's relationship with the target rather than its overall contribution to reducing impurity.

Bagging : 
Random Forest uses bagging (Bootstrap Aggregating), where each tree is trained on a random sample of the training data selected with replacement. At every split, only a random subset of approximately √(number of features) is considered for splitting. The predictions from all trees are then combined, which reduces the variance of the model and makes it more stable and less prone to overfitting than a single deep decision tree.

4b.Feature ABlation :
The five least important features identified by the Random Forest were City_Ernakulam, City_Bhopal, City_Coimbatore, City_Chandigarh, and City_Mumbai. The full model achieved a ROC-AUC of 0.9408, while the reduced model achieved a ROC-AUC of 0.9388. Since the AUC decreased only slightly, these features contributed very little to the model's performance. This suggests that a simpler model with fewer features can reduce inference cost and maintenance while maintaining similar performance, provided the small reduction in AUC is acceptable for the application.

Task 5- Cross Fold 
Model                       Mean AUC   Std AUC
0       Logistic Regression  0.943248  0.005644
1  Controlled Decision Tree  0.834038  0.012608
2             Random Forest  0.926723  0.010624
3         Gradient Boosting  0.929707  0.007983
The models were evaluated using 5-fold Stratified Cross-Validation with ROC-AUC as the evaluation metric. Cross-validation provides a more reliable estimate of generalization performance because the model is trained and evaluated on multiple train-test splits instead of a single split. This reduces the impact of random variation in the data and gives a more robust measure of how well the model is expected to perform on unseen data.

Task 6 - Hyperparmeter

A GridSearchCV was performed using a Random Forest pipeline with 18 hyperparameter combinations (3 × 3 × 2). Using 5-fold cross-validation, a total of 90 model fits were evaluated. The best parameters were n_estimators = 200, max_depth = None, and min_samples_leaf = 1, achieving a best cross-validated ROC-AUC of 0.9587. Grid Search evaluates every possible parameter combination to find the best model but can be computationally expensive. In contrast, Randomized Search evaluates only a random subset of parameter combinations, making it faster but with no guarantee of finding the optimal combination.

