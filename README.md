# Wine Quality Prediction Model
## Overview

**Purpose**

The purpose of the analysis is to determine if the __________ model can accurately predict the quality of a wine based on physiochemical properties, assigning wines a low, average, or high score.

**Dataset**

Datasets cited in this project are from the UC Irvine Machine Learning Repository Website.
https://archive.ics.uci.edu/dataset/186/wine+quality

The data points in the datasets are based on data from the 2009 paper, Modeling wine preferences by data mining from physicochemical properties.
https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377?via%3Dihub

The combined dataset of 6,497 data points, used to perform this analysis is comprised of columns including the following variables:
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
### Data Preprocessing
1. Data Preparation: Import files, establish DataFrames, and evaluate the columns and features.
- Originally, 2 separate datasets existed, one for red wine with 1,599 data points and the other for white wine with 4,898 data points. As part of the data preparation process, we combined these datasets, totaling 6,497 data points. 
2. 
### Compiling, Training, and Evaluating the Model
1.


		
## Results



## Summary



## References

Cortez, Paulo, et al. “Modeling Wine Preferences by Data Mining from Physicochemical Properties.” _Decision Support Systems_, vol. 47, no. 4, Nov. 2009, pp. 547–553, https://doi.org/10.1016/j.dss.2009.05.016.

Diehm, Jan, and Lars Verspohl. “Wine & Math: A Model Pairing.” _The Pudding_, 2021, pudding.cool/2021/03/wine-model/. Accessed 5 Sept. 2024.

Parrish, Michael. “What Are Wine Varietals? Comprehensive Guide to Grape Varieties | the Tasting Alliance | the Tasting Alliance.” _The Tasting Alliance_, 14 Mar. 2024, thetastingalliance.com/what-are-wine-varietals-comprehensive-guide-to-grape-varieties/. Accessed 5 Sept. 2024.

The Wine Wankers. “A Great Collection of Wine Infographics.” _The Wine Wankers_, 12 Feb. 2014, winewankers.com/2014/02/12/a-great-collection-of-wine-infographics/. Accessed 5 Sept. 2024.

Theethat Anuraksoontorn. “Wine Quality Prediction with Python - Analytics Vidhya - Medium.” _Medium_, Analytics Vidhya, 21 Sept. 2020, medium.com/analytics-vidhya/wine-quality-prediction-with-python-695939d34d87. Accessed 5 Sept. 2024.

Wine Paths. “The Ultimate Beginners Guide to Wine.” _Www.winepaths.com_, www.winepaths.com/articles/editorial/wine-guide/the-ultimate-beginners-guide-to-wine. Accessed 5 Sept. 2024.

> Written with [StackEdit](https://stackedit.io/).
