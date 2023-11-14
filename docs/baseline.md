# Baseline Models

In the provided notebook, we implemented baseline models for predicting goal outcomes in hockey games. The notebook begins by importing necessary libraries, setting up directories for data, models, and figures, and initializing Comet experiments for experiment tracking. It then loads the dataset, preprocesses it by selecting desired features, and splits it into training and validation sets. Logistic regression models are trained separately for shot distance, shot angle, and a combination of both features. The notebook includes visualizations such as confusion matrices, ROC curves, goal rate, cumulative percentage of goal, and calibration curves for each model. Random predictions are also generated as a baseline. The notebook concludes with logging relevant metrics and saving the trained models, confusion matrices, and metrics on Comet for further analysis. The models and experiments are tagged appropriately, indicating their baseline nature and the features used in training.

## Task 1 - Unveiling the Accuracy Paradox

<figure>
  <img style="float: left;" src="readme-imgs/hist.png" width="41%" height: auto>
</figure>

The evaluation results indicate a high accuracy of 91% on the validation set, but upon closer inspection of the precision, recall, and f1-score for label 1 (goal), it becomes apparent that the model is unable to correctly predict instances of this class, yielding zeros in these metrics. The issue arises from the significant class imbalance in the dataset, where label 0 (non-goal) vastly outnumbers label 1. With 276,782 samples for label 0 and only 29,032 samples for label 1, the model might be biased towards predicting the majority class, achieving high accuracy due to the dominance of label 0 in the dataset. However, this high accuracy is misleading, as the model struggles to capture the minority class (label 1).


## Task 2 - Evaluating Baselines

<img src="readme-imgs/task2.png" width="90%" height: auto>

Baseline Model - Distance: <a href="[url](https://www.comet.com/ift6758-b09-project/ift6758-project-milestone2/fd6f683bf9324bc4aafe732516e9ed38)">(Link)</a>

Baseline Model - Angle: <a href="[url](https://www.comet.com/ift6758-b09-project/ift6758-project-milestone2/066eb71923294143887d23136514beb5)">(Link)</a>

Baseline Model - Distance + Angle: <a href="[url](https://www.comet.com/ift6758-b09-project/ift6758-project-milestone2/90775941400f48689503c7bacdc0ff09)">(Link)</a>

As we can see, the simplest baseline, random prediction, acts exactly the same as random models, having 50 percent area under the curve in the ROC plot and steady in goal rate and calibration plots. On the other hand, we have noticed slight improvement from the angle model to the distance model and then from there to the model using a combination of them showing that as we increase the number of features, we encounter better performance in evaluation.