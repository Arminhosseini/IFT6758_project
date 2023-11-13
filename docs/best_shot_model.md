## 6. Best shot model

### a. Data pre-processing

In this section, we will provide step-by-step our data pre-processing step.

1. Add more feature

In this section, we will add the new feature called `attacking_zone_shot`. This feature indicates whether a shot took place within the opposing team's attacking zone. We choose to analyze this feature based on the observation that the shot in the attacking zone has high chance to become a goal.

After analyzing, we've discerned that 94.5% of shots occurred within the attacking zone, while the remaining 5% happened outside this zone.

2. Feature selection - correlation

In this section, we will check the correlation value between pairs of features. If the correlation between two features is larger than the specific `threshold`, we will remove it. In this experiment, we set the threshold is 0.9.

After analyzing, we remove features `['period']`.

3. Feature selection - mutual information

On this section, we will use mutual information to identify the relationship between features and the target variable. The intuition behind this experiment is that higher mutual information indicates stronger predictive power.

After calculating the mutual information between each feature and target variables. We got the result
![](/images/best_shot_model/mutual_information.jpg)

According to the experiment, we will set the threshold below **0.01**. In other words, If the mutual information between each feature and target variable is smaller than 0.01, we will remove it. 

After analyzing, we choose feature:
`['x-coordinate', 'y-coordinate', 'shot_distance', 'angle', 'isEmptyNet', 'n_friend', 'n_oppose']`

4. Balance dataset

Because the distribution of label in the training set is imbalanced. We apply the over-sample to add more samples to the minority class.

### b. Machine learning model grid search

In this section, we run two type of models: decision tree and KNN. For the purpose of this experiment, we just run the simple grid search experiment and leave the complicated work later. The grid hyperparameter tuning on each model:
- Grid search decision tree:
```
param_grid_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [5, 10],
}
```

- Grid search KNN:
```
param_grid_knn = {
    'n_neighbors': [2, 3],
    'weights': ['uniform', 'distance']
}
```
We create four figures (ROC/AUC curve, goal rate vs probability percentile, cumulative proportion of goals vs probability percentile, and the reliability curve) to both ML model: decision tree and knn

1. The ROC/AUC curve

![](/images/best_shot_model/roc.jpg)


2. The goal rate vs probability percentile, cumulative proportion of goals vs probability percentile

![](/images/best_shot_model/goal_rate_cum.jpg)


3. The reliability curve

![](/images/best_shot_model/calibration.jpg)


In addition, we also inspect the accuracy and f1 score on the validation set:

|                | Accuracy | F1 score |
| -------------- | -------- |--------  |
| Decision Tree  |  90.86%  | 0.73     |
| KNN            |  85.97%  | 0.59     | 

In summary, the decision tree is the best model we can build.

### c. Log the models to the experiments on comet.ml




