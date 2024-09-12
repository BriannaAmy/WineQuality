# Wine Quality Prediction Model

## Overview
### Background


### Purpose

The purpose of this project is to provide wine quality predictions based on physiochemical properties using neural network and random forest models. This analysis aims to improve machine learning capabilities by recategorizing quality measurements to reduce the number of outlying data points and increasing available data by combining datasets.

### Datasets

Datasets cited in this project are from the UC Irvine Machine Learning Repository Website.

https://archive.ics.uci.edu/dataset/186/wine+quality

The data points in the datasets are based on data from the 2009 paper, *Modeling Wine Preferences by Data Mining from Physicochemical Properties*.

https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377?via%3Dihub

The repository included 2 datasets, one for red wine and another for white wine. The red wine dataset contained 1,599 data points and the white wine dataset contained 4,898 data points. The data was collected from the Vinho Verde region in the northwest of Portugal. Each wine was analyzed, and assessed for quality between 2004 and 2007 by the Viticulture Commission of the Vinho Verde Region.

The datasets included the following columns/variables:

- **Fixed Acidity**: non-volatile acids found in wine which are tartaric, malic, citric, and succinic.

- **Volatile Acidity**: the acidic elements of a wine that are gaseous; aroma

- **Citric Acid**: weak organic acid, often used as a natural preservative or additive to food/drink; sour taste

- **Residual Sugar**: the amount of sugar remaining after fermentation stops

- **Chlorides**: the amount of salt in the wine

- **Free Sulfur Dioxide**: amount of SO2; used to prevent oxidation and microbial growth

- **Total Sulfur Dioxide**: amount of free and bound forms of S02S02, evident in the nose and taste of wine; mostly undetectable

- **Density**: dependent on percent of alcohol and sugar content

- **pH**: used to measure ripeness in relation to acidity. Low pH wines tastes tart, while higher pH wines are more susceptible to bacterial growth. Most wines fall around 3 or 4

- **Sulphates**: wine additive that contributes to sulfur dioxide gas levels; acts as an antimicrobial and antioxidant

- **Alcohol**: percent alcohol content of the wine

- **Quality**: overall score between 0 and 10

### Methods and Technologies
* Postgres SQL
* pandas
* numpy
* train_test_split from sklearn
* StandardScaler from sklearn
* tensorflow
* pathlib from path
* RandomForestClassifier from sklearn
* confusion_matrix from sklearn
* accuracy_score from sklearn
* classification_report from sklearn
* matplotlib
* RandomOverSampler from imblearn
* seaborn
* pymongo
* keras


## Data & Modeling Steps

#### ETL
1. Import dependencies.
2. Read the white and red wine CSV file.
3. Retrieve index types and column names.
4. Place the columns and corresponding values into a list.
5. Create a data frame for white and red wine.
6. Add a "Type" column to both data frames (0 represents white wine and 1 represents red wine). Ensure the "Type" column is executing properly.
7. Group white wine quality into 3 different groups. (0 - Bad, 1 - Average, 2 - Great). Create a "quality_category" column and ensure the column is added correctly.
8. Repeat Step 7 for red wine.
9. Create a third data frame, merging white and red wine data frames.
10. View summary statistics for each data frame.
11. Add an ID column to serve as the primary key for each data frame.
12. Load into Postgres SQL. Create database connection for Postgres SQL.
13. Define and execute SQL query. Create data frame and display—close connection.
14. Read in the data frame from the Postgres SQL database and split the data frame based on the "Type" column.
15. Write white wine to CSV and red wine to CSV.
#### Initial Approach:
##### Neural Network Deep Learning Model
1. Import dependencies.
2. Read and merge CSV files. Show data frame.
3. Split data into features and target array. Split data into testing and training datasets.
4. Create StandardScaler instances. Fit and scale the data. 
5. Define the model- first, second, and output later. Review the structure of the model.
6. Compile model, converting y_train to categorical.
7. Train the model.
8. Evaluate the model using test data.
##### Random Forest Model
1. Import dependencies.
2. Create RandomForestClassifier.
3. Fit the model and make predictions using the testing data.
4. Calculate the confusion matrix and accuracy score.
5. Random Forests automatically calculate feature importance. We then sort the features by their importance.
6. Visualize the features by importance.
#### Optimized Approach:
1. Import dependencies.
##### White Wine, Red Wine, and Combine Neural Network Models
1. Read CSV. Display information and value counts.
2. Define a function to categorize the quality and apply it to the "quality_category" column.
3. Remove quality target from features data.
4. Split into training and testing datasets.
5. Apply RandomOverSampler and review new class distribution.
6. Create StandardScaler instances. Fit and Scale the data.
7. Define sequential deep learning model. Change output layer units to 3 to match the number of unique values in the target variables.
8. Compile sequential mode and customize metrics. Use categorical_crossentropy. Convert y_train to categorical.
9. Train the model.
10. Evaluate the model.
##### Combined Data Random Forest Model
1. Re-create the combined data frame to include the quality column.
2. Define features set.
3. Define target vector.
4. Apply RandomOverSampler and review new class distribution.
5. Split into training and testing sets.
6. Create StandardScaler instance. Fit and Scale the data.
7. Create and train RandomForestClassifier.
8. Fit the model and make predictions using the testing data.
9. Calculate the confusion matrix.
10. Define labels for the 3 classes.
11. Calculate the accuracy score.
12. Random Forests automatically calculate feature importance. We then sort the features by their importance.
##### Visualizations
1. Feature Importance.
2. Correlation Matrix visualization.
3. Histogram visualization.
4. Alcohol Distribution Plot visualization.
5. Alcohol and Quality Box Plots visualizations.
6. Model Accuracy Comparison visualizations.
## Results
### Neural Network Model
***White Wine***
* Optimized Model: accuracy: 0.820 - loss: 0.399

***Red Wine***
* Optimized Model: accuracy: 0.868 - loss: 0.308

***Combined***
* Initial Model (w/o Quality Category): accuracy: 0.556 - loss: 1.037
* Optimized Model (w/ Quality Category): accuracy: 0.827 - loss: 0.391
### Random Forest Model
***Combined***
* Initial Model (w/o Quality Category): 0.689
* Optimized Model (w/ Quality Category): 0.673

## Summary

## References

Cortez, Paulo, et al. “Modeling Wine Preferences by Data Mining from Physicochemical Properties.” _Decision Support Systems_, vol. 47, no. 4, Nov. 2009, pp. 547–553, https://doi.org/10.1016/j.dss.2009.05.016.

Diehm, Jan, and Lars Verspohl. “Wine & Math: A Model Pairing.” _The Pudding_, 2021, pudding.cool/2021/03/wine-model/. Accessed 5 Sept. 2024.

Parrish, Michael. “What Are Wine Varietals? Comprehensive Guide to Grape Varieties | the Tasting Alliance | the Tasting Alliance.” _The Tasting Alliance_, 14 Mar. 2024, thetastingalliance.com/what-are-wine-varietals-comprehensive-guide-to-grape-varieties/. Accessed 5 Sept. 2024.

Schull, Trista. “How Wine Is Made,” _Witches Falls Winery_, 7 Sept. 2022, witchesfalls.com.au/blogs/news/how-wine-is-made. Accessed 6 Sept. 2024.

The Wine Wankers. “A Great Collection of Wine Infographics,” _The Wine Wankers_, 12 Feb. 2014, winewankers.com/2014/02/12/a-great-collection-of-wine-infographics/. Accessed 5 Sept. 2024.

Theethat Anuraksoontorn. “Wine Quality Prediction with Python - Analytics Vidhya - Medium.” _Medium_, Analytics Vidhya, 21 Sept. 2020, medium.com/analytics-vidhya/wine-quality-prediction-with-python-695939d34d87. Accessed 5 Sept. 2024.

Wine Paths. “The Ultimate Beginners Guide to Wine.” _Www.winepaths.com_, www.winepaths.com/articles/editorial/wine-guide/the-ultimate-beginners-guide-to-wine. Accessed 5 Sept. 2024.

> Written with [StackEdit](https://stackedit.io/).
