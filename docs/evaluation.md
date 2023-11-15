## 7. Evaluation

Overall, our model perform well on both regular and playoff game, which mean that the generalization ability of our model is good.

### 7.1. Evaluate on regular season

On the logistic regression, the features combining distance and angle is better than each feature seperately. 

The performance of the decision tree model is better than the baseline logistic regressions. One reason to explain that we apply more features and we balance the dataset. Moreover, the decision tree is quite suitable for this type of tabular data. 

- Best?

Our models perform with the same ROC, goal rate, cumulative proportion, during the validation and testing set.

The ROC of different models on the regular season:
|        Model                     | ROC score |
| --------------                   | --------  |
| Logistic regression (dist)       |   0.69  |
| Logistic regression (angle)      |   0.56  |
| Logistic regression (dist+angle) |   0.71  |
| XGBoost                          |      |
| Decision Tree                    |   0.75  |


### 7.2. Evaluate on playoff game

The ROC of different models on the playoff game:
|        Model                     | ROC score |
| --------------                   | --------  |
| Logistic regression (dist)       |   0.68  |
| Logistic regression (angle)      |   0.57  |
| Logistic regression (dist+angle) |   0.70  |
| XGBoost                          |           |
| Decision Tree                    |   0.75  |

There is slightly difference between ROC on the regular season and playoff games.

The performance of the trained decision tree is acceptable, and higher than the baseline logistic regression.

Best?