# Wine Quality Prediction Model
## Overview

**Purpose**

The purpose of this project is to provide improved wine quality predictions based on physiochemical properties using neural network and random forest models. In this model, we recategorized quality measurement data and combined datasets as a means to reduce outlier distortion and improved machine learning capabilities.

**Dataset**

Datasets cited in this project are from the UC Irvine Machine Learning Repository Website.
https://archive.ics.uci.edu/dataset/186/wine+quality

The data points in the datasets are based on data from the 2009 paper, Modeling wine preferences by data mining from physicochemical properties.
https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377?via%3Dihub

The repository included 2 datasets, one for red wine and another for white wine. The red wine dataset contained 1,599 data points and the white wine dataset contained 4,898 data points. The data was collected from the Vinho Verde region in the northwest of Portugal. Each wine was analyzed, and quality assessed between 2004 and 2007 by the Viticulture Commission of the Vinho Verde Region.

The datasets included the following columns / variables:
- **Fixed Acidity**: non-volatile acid found in wine which are tartaric, malic, citric, and succinic.
- **Volatile Acidity**: the acidic elements of a wine that are gaseous; aroma
- **Citric Acid**: weak organic acid, often used as a natural preservative or additive to food/drink; sour taste
- **Residual Sugar**: amount of sugar remaining after fermentation stops
- **Chlorides**: the amount of salt in the wine
- **Free Sulfur Dioxide**: amount of SO2; used to prevent oxidation and microbial growth
- **Total Sulfur Dioxide**: amount of free and bound forms of S02S02, evident in the nose and taste of wine; mostly undetectable
- **Density**: dependent on percent of alcohol and sugar content
- **pH**: used to measure ripeness in relation to acidity. Low pH wines tastes tart, while higher pH wines are more susceptible to bacterial growth. Most wines fall around 3 or 4
- **Sulphates**: wine additive that contributes to sulfur dioxide gas levels; acts as an antimicrobial and antioxidant
- **Alcohol**: percent alcohol content of the wine
- **Quality**: overall score between 0 and 10

**Methods and Technologies Utilized**
* **Scikit-learn**
train_test_split, StandardScaler, RandomForestClassifier, confusion_matrix, accuracy_score, classification_report
* **Python**
Pandas, Matplotlib, Pathlib
* **Imblearn**
RandomOverSampler
* **TensorFlow**
* **Seaborn**

## Data & Modeling Approach
**Perform Initial imports for all datasets**

`import pandas as pd`

`from pathlib import Path`

`from sklearn.ensemble import RandomForestClassifier`

`from sklearn.preprocessing import StandardScaler`

`from sklearn.model_selection import train_test_split`

`from sklearn.metrics import confusion_matrix, accuracy_score, classification_report`

`%matplotlib inline`

`from sklearn.model_selection import train_test_split`

`from sklearn.preprocessing import StandardScaler`

`from sklearn.metrics import accuracy_score`

`import tensorflow as tf`

`from imblearn.over_sampling import RandomOverSampler`

**White Wine Dataset**
*4,898 data points*
1. Read csv file. Convert to DataFrame. Examine info and value counts.
2. Define a function to categorize the quality:
`def categorize_quality(quality):`
    `if 1 <= quality <= 3:
        return '0'
    elif 4 <= quality <= 6:
        return '1'
    elif 7 <= quality <= 9:
        return '2'`
3. Apply the function to create the 'quality_category' column and verify changes.
4. Drop 'quality' column.
5. Review 'quality_category' value counts.
6. Remove quality target from features data. Split training/test datasets.
7. Apply `RandomOverSampler` and review new class distribution.
8. Preprocess numerical data for neural network. Create `StandardScaler` instances. Fit and scale the data. Review the shape of the scaled data to check for inconsistencies.
9. Define the deep learning model. Change output layer units to 3 to match the number of unique values in the target variable.
`nn_model = tf.keras.models.Sequential()`

`nn_model.add(tf.keras.layers.Dense(units=16, activation="relu", input_dim=12))`

`nn_model.add(tf.keras.layers.Dense(units=16, activation="relu"))`

`nn_model.add(tf.keras.layers.Dense(units=3, activation="softmax"))`

11. Compile the Sequential model together and customize metrics. Convert 'y_train' to categorical. Train the model.
12. Evaluate the model using the test data.

**Red Wine Dataset**
*1,598 data points*
Repeat Steps 1-11 on previous dataset, using Red Wine Dataset. 

**Combined Datasets**
*6,496 data points*
**Neural Network Model**
1. Convert white and red DataFrames into Combined DataFrame. Examine info and value counts.
2. Repeat Steps 2-11 of previous datasets.
**Random Forest Model**
3. Drop 'quality_category'.
4. Define features set. Define target vector. Review 'quality' value counts
5. Apply `RandomOverSampler` and review new class distribution.
6. Split into Train and Test sets. Create `StandardScaler` instance. Fit and scale the data. 
7. Assuming 'target_variable' is the name of your target column, split data into training and testing sets. Create and train the `RandomForestClassifier`.
8. Fit the model. 
9. Make predictions using the testing data.
10. Changed 'model' to 'rf_model'. Assuming 'rf_model' is your trained classification model and 'X_test', 'y_test' are your test datasets.
11. Calculated confusion matrix.
12. Defined labels for the 3 classes.
13. Calculated accuracy score.
14. Display Feature Importances:
    
`[(0.1228541834756794, 'alcohol'),`

 `(0.10213457013517334, 'volatile acidity'),`
 
` (0.10185320716189483, 'density'),`

` (0.09113809109468214, 'total sulfur dioxide'),`

` (0.08677057522512018, 'sulphates'),`

` (0.08668949352072257, 'chlorides'),`

 `(0.08611019795650059, 'free sulfur dioxide'),`
 
 `(0.08401862460115347, 'residual sugar'),`
 
 `(0.08355183861811914, 'pH'),`
 
` (0.0803368568620962, 'citric acid'),`

` (0.07454236134885828, 'fixed acidity')]`

		
## Results
**White Wine**
* `39/39 - 0s - 5ms/step - accuracy: 0.9976 - loss: 0.0155`

* `Loss: 0.015487316064536572, Accuracy: 0.9975510239601135`

**Red Wine**
* `13/13 - 0s - 15ms/step - accuracy: 0.7500 - loss: 0.6183`

* `Loss: 0.6182871460914612, Accuracy: 0.75`

**Combined**
* ***Neural Network Model***
	`51/51 - 0s - 5ms/step - accuracy: 0.9415 - loss: 0.1556`

	`Loss: 0.15556824207305908, Accuracy: 0.9415384531021118`

* ***Random Forest Model***
	`Accuracy Score : 0.41846153846153844`

## Summary
The white wine model provided the most sufficient, valid data output with a loss of approximately 0.016 (1.6%) and accuracy of 0.98 (98%). This illustrates that errors occurred in less than 2% of the iterations performed by this model. It can also be stated that a majority of predictions made by this model were correct.

The red wine model provided less sufficient, valid data output with a loss of approximately 0.62 (62%) and accuracy of 0.75 (75%). This illustrates that errors occurred in more than half of the iterations performed by this model. It can also be stated that a majority of predictions made by this model were correct.

The combined model provided expected performance results, with a loss of approximately 0.16 (16%), and accuracy of 0.94 (94%). This illustrates that by combining the datasets, the percentage of correct predictions only decreased by 4% and the model's iteration errors fell into a reasonable range of 15%.

After reviewing the results, it is evident that combining the datasets provided more data for the learning capabilities of our models. This led to more realistic loss and accuracy results. Another key factor in the success of this project analysis is the recategorization of the quality data. By grouping the data into 3 categories versus the original 11 categories, the model was able to eliminate outliers that led to decreased accuracy scores achieved by similar models.


## References

Cortez, Paulo, et al. “Modeling Wine Preferences by Data Mining from Physicochemical Properties.” _Decision Support Systems_, vol. 47, no. 4, Nov. 2009, pp. 547–553, https://doi.org/10.1016/j.dss.2009.05.016.

Diehm, Jan, and Lars Verspohl. “Wine & Math: A Model Pairing.” _The Pudding_, 2021, pudding.cool/2021/03/wine-model/. Accessed 5 Sept. 2024.

Parrish, Michael. “What Are Wine Varietals? Comprehensive Guide to Grape Varieties | the Tasting Alliance | the Tasting Alliance.” _The Tasting Alliance_, 14 Mar. 2024, thetastingalliance.com/what-are-wine-varietals-comprehensive-guide-to-grape-varieties/. Accessed 5 Sept. 2024.

Schull, Trista. “How Wine Is Made,” _Witches Falls Winery_, 7 Sept. 2022, witchesfalls.com.au/blogs/news/how-wine-is-made. Accessed 6 Sept. 2024.

The Wine Wankers. “A Great Collection of Wine Infographics,” _The Wine Wankers_, 12 Feb. 2014, winewankers.com/2014/02/12/a-great-collection-of-wine-infographics/. Accessed 5 Sept. 2024.

Theethat Anuraksoontorn. “Wine Quality Prediction with Python - Analytics Vidhya - Medium.” _Medium_, Analytics Vidhya, 21 Sept. 2020, medium.com/analytics-vidhya/wine-quality-prediction-with-python-695939d34d87. Accessed 5 Sept. 2024.

Wine Paths. “The Ultimate Beginners Guide to Wine.” _Www.winepaths.com_, www.winepaths.com/articles/editorial/wine-guide/the-ultimate-beginners-guide-to-wine. Accessed 5 Sept. 2024.

> Written with [StackEdit](https://stackedit.io/).
