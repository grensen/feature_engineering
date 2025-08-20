# Feature Engineering Intuition

## One Perceptron (Neural Network Playground)

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/playground_linear2.gif?raw=true" alt=""/>
</p>

Why feature engineering? The XOR classification problem is a famous example of perceptron limitations and is historically related to the first AI winter. The problem is that a standard perceptron can only separate data with a straight line, but the data shows a checkerboard pattern, so a single linear layer cannot solve it.
However, a neural network with one hidden layer (inputs-hidden1-outputs) can solve the problem. You can experiment with this on the TensorFlow Playground: https://playground.tensorflow.org/

## Extended Features

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/intuition_fully.png?raw=true" alt=""/>
</p>

The solution for a single layer (inputs-outputs) involves feature engineering, where we create new features from the original inputs. For example, creating a new feature like X1 * X1. With the right features, the problem can be almost solved with only one layer.

## The Right Feature

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/intuition_perfect.png?raw=true" alt=""/>
</p>

With the right feature, the solution becomes perfect, and you may not even need to train the linear layer. For example, consider the feature X1 * X2. If X1 and X2 have the same sign (both positive or both negative), their product is positive. If their signs are different, the result will be negative (e.g., 1 * -1 = -1).
This makes the data perfectly separable: if the new feature's value is greater than 0, it's one class, and if it's less than 0, it's the other.

## A Simple Spiral and Two Classes

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/curved_linearity.gif?raw=true" alt=""/>
</p>

This is from an old animation where I used new features to solve a spiral classification problem with only a single layer. These features achieve something that was previously impossible: they allow a linear model to separate the spiral data, effectively "bending" the decision boundary.


## Complex Spirals With 10 Classes

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/complex_spiral_ultra.png?raw=true" alt=""/>
</p>

The 10-class spiral is a very difficult problem, but a neural network combined with feature engineering was able to classify the data with high accuracy.


## Optimization Dataset HiLo1

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/hilo1.png?raw=true" alt=""/>
</p>

Measuring feature importance is also critical. My approach is unconventional: it analyzes the features of the data points with the highest and lowest target values (the "Error" label).
The blue and gold balls show the mean and standard deviation of each feature for these two extreme groups. A feature is considered important if its statistics (mean and std) are significantly different between the "high-error" and "low-error" groups. This importance is visualized by the width of each feature.

## Optimization: HiLo4

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/simple_mean_var_test_hilo4.png?raw=true" alt=""/>
</p>

This is the intuition behind the "HiLo" evaluation. The method gives more weight to the data points with the most extreme labels. For example, HiLo200 weights the impact of the highest label 200 times, the second-highest 199 times, and so on.
Similarly, HiLo4 analyzes the four best and four worst labels, ensuring that the most extreme values have the greatest influence. Unfortunately, the "ball" visualization for this is not yet accurate, but the width and percentage for each feature still show the final, approximated impact.

## Feature Importance With an Optimization Dataset (200)

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/feature_importance.gif?raw=true" alt=""/>
</p>

The balls can show only the latest mean and std. But to test the range from HiLo 1-200, where the size of the dataset is also 200, this animation comes close to what I would expect as a neat result to reflect the importance of each feature.

## Create a New Feature

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/create_feature.gif?raw=true" alt=""/>
</p>

Now, let’s do the real stuff. But before that, think about this: normalization is not only a basic machine learning skill, it is also the foundation for this visualization technique. So when I look at this, I must think of values between 0 and 1 for each feature. HiLo is set to 100 on a dataset of size 200.
The new feature (F3*F3+F2) has a much stronger importance score of 25.2%, but even more important to me is the visual proof that this new feature arranges its line colors almost right from blue to gold.

## Banknote Dataset (1079)

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/banknote_feature.png?raw=true" alt=""/>
</p>

The banknote binary classification dataset is another nice problem where label 0 represents genuine banknotes and label 1 represents forged ones. It was possible to find very simple new features to separate both classes for all examples in the dataset, which is really impressive to me.

## Banknote Demo

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/banknote_feat_eng_demo_fixed.png?raw=true" alt=""/>
</p>

I created a demo in C# to test this visually developed prediction model, [Code can be found here](https://github.com/grensen/feature_engineering/blob/main/banknote_feat_eng_demo.cs). The prediction weight of 1.37 is a direct result of this visual work.
It's also possible to convert this value to a percentage. For example, the highest score for a genuine banknote could be set as the 100% baseline. This would allow us to measure how strongly a banknote is flagged as "forged," especially if its score goes far above 100%.
There is little room to improve the prediction. The only way forward seems to be for counterfeiters to "crack" this system, which would force an engineer to improve the model again. 

## Concrete Compressive Strength (1005)

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/concrete_feature.png?raw=true" alt=""/>
</p>

The concrete compressive strength dataset is another real-life example of a regression problem. Here, the most important feature by far is Cement, which has a strong positive correlation to the label. High cement amounts lead to strong concrete and vice versa, but not always.
The F9 label is the result of experimentation. It seems that a combination of many feature transformations (I don't know what I really did here) can lead to much stronger features that also seem more reliable than the standard features of the dataset.
But be careful. Things can change, and sometimes they must. Each dataset is a partial insight into the real distribution, and more data can change the picture. The model will reflect that. Just give it a try. We can do much more with our features than I showed here. Watch out the further impressions.

## Bad Logic: Feature Process Foward (F5 + F5 + F5 + F6)

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/create_feature_forward.gif?raw=true" alt=""/>
</p>

## Good Logic: Feature Process Backward (F6 + F5 + F5 + F5)

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/create_feature_reverse.gif?raw=true" alt=""/>
</p>

## Programmatic Feature Engineering

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/feature_programmatic_demo.png?raw=true" alt=""/>
</p>

## The Entire Hardcoded Model

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/2025-feature_engineering_model.png?raw=true" alt=""/>
</p>

## Feature Engineering Class Functions (Predict or Transform X Array)

<p align="center">
  <img src="https://github.com/grensen/feature_engineering/blob/main/figures/feature_engineering_class.png?raw=true" alt=""/>
</p>
