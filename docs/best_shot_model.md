## 6. Best shot model

### a. Data pre-processing

In this section, we will provide step-by-step our data pre-processing step, including: add feture, feature selection using correlation, feature selection using mutual information, and balance dataset.

#### i. Add feature

In this section, we will add the new feature called `attacking_zone_shot`. This feature indicates whether a shot took place within the opposing team's attacking zone. We choose to analyze this feature based on the observation that the shot in the attacking zone has high chance to become a goal.

After analyzing, we've discerned that 94.5% of shots occurred within the attacking zone, while the remaining 5% happened outside this zone.

#### ii. Feature selection - correlation

In this section, we will check the correlation value between pairs of features. If the correlation between two features is larger than the specific `threshold`, we will remove it. In this experiment, we set the threshold is 0.9.

After analyzing, we remove features `['period']`.

#### iii. Feature selection - mutual information

On this section, we will use mutual information to identify the relationship between features and the target variable. The intuition behind this experiment is that higher mutual information indicates stronger predictive power.

After calculating the mutual information between each feature and target variables. We got the result
![](/images/best_shot_model/mutual_information.jpg)

According to the experiment, we will set the threshold below **0.01**. In other words, If the mutual information between each feature and target variable is smaller than 0.01, we will remove it. 

After analyzing, we choose feature:
`['x-coordinate', 'y-coordinate', 'shot_distance', 'angle', 'isEmptyNet', 'n_friend', 'n_oppose', 'last_event_type', 'is_rebound', 'attacking_zone_shot']`

#### iv. Balance dataset

Because the distribution of label in the training set is imbalanced. We apply the over-sample to add more samples to the minority class.

### b. Machine learning model grid search

In this section, we run two type of models: decision tree and logistic regression with the preprocessed dataset. For the purpose of this experiment, we just run the simple grid search experiment and leave the complicated work later. The grid hyperparameter tuning on each model:
- Grid search decision tree:
```
param_grid_tree = {
    "tree__criterion": ['gini', 'entropy'],
    "tree__max_depth": [5, 10],
    "tree__min_samples_leaf": [5, 10]
}
```

- Grid search logistic regression:
```
param_grid_linear = {
    "linear_clf__penalty": ['l1', 'l2'],
    "linear_clf__C": [0.1, 0.01]
}

```
We create four figures (ROC/AUC curve, goal rate vs probability percentile, cumulative proportion of goals vs probability percentile, and the reliability curve) to both ML model: decision tree and knn

#### i. The ROC/AUC curve

![](/images/best_shot_model/roc.jpg)


#### ii. The goal rate vs probability percentile

![](/images/best_shot_model/goal_rate_cum.jpg)


#### iii. The cumulative proportion of goals vs probability percentile
![](/images/best_shot_model/goal_cumulative_proportion.jpg)


#### iii. The reliability curve

![](/images/best_shot_model/calibration.jpg)


In addition, we also inspect the accuracy and f1 score on the validation set:

|                | Accuracy | F1 score |
| -------------- | -------- |--------  |
| Decision Tree  |  90.7%  | 0.738     |
| Logistic regression  |  90.3%  | 0.731     | 

In summary, the decision tree is the best model we can build.

### c. Log the models to the experiments on comet.ml

We have add the trained decision tree and trained logistic model into our Model Registry. 

You can download it with those links:
- Decision tree model: https://www.comet.com/api/registry/model/item/download?modelItemId=4EKykFYfUY9Izp5mksrUcvrn8
- KNN model: https://www.comet.com/api/registry/model/item/download?modelItemId=4EKykFYfUY9Izp5mksrUcvrn8

We also add the evaluation metrics to the Comet experiment. We add 3 metrics, including: Accuracy, ROC, and confusion matrix. 
You can access each individual experiment with the tag `best_shot_model` for more details. 

In case you want to check the metrics each experiment in a tabular format, you can check via this link: 
https://www.comet.com/ift6758-b09-project/ift6758-project-milestone2/view/new/experiments



