# Wine_Quality
Using supervised machine learning techniques LASSO and random forest to predict quality of wine 
Wine Quality: Predicting Ratings Using Supervised Learning Techniques

# Introduction
The quality of wine is of utmost importance to consumers, producers and marketers alike, which makes this project rather valuable. The goal of this project is to predict wine quality based on various physicochemical properties like alcohol content, pH levels, acidity and many more. This prediction is done by employing supervised learning techniques namely, LASSO Regression and Random Forest, while using Linear Regression as a base model for the purpose of comparison.

# The Dataset
The data sets being used can be accessed at UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/wine+quality ). The dataset contains 12 variables that were recorded for 1,599 observations of red wine and 4,898 observations of white wine. Using this data, regression and random forest models were built to predict the impact of predictor variables on the target variable that is quality. If applied to real life situations, this analysis can help producers and businesses come up with informed business strategies to increase sales. 

# Data Cleaning and Exploratory Data Analysis
### 1. Delimiting data
The data when downloaded was not in the correct format.
The data was delimited in MS Excel in order to separate it into columns for better readability and to easily work on the data. 

### 2. Data Preparation
Both, red and white wine data sets were successfully loaded onto the R workspace for further analysis. As part of preliminary data cleaning, the data set was checked for missing values.
Data was cleaned before moving forward, as an impure data set would introduce bias in the models which is not desirable for accuracy of our models. 
For creating a more unified dataset for comprehensive analysis and modeling, the two datasets were merged. By merging the datasets, a new wine_type column was added to treat wine type as a categorical variable to model any potential differences in wine quality based on wine type. Merging the datasets allows for a more holistic approach to our models. Since the dataset is not very large, merging the datasets gives the model enough data to work with. It allows for a more reliable model building.
 wine_type was converted to factor for the following reasons:

a) Model Interpretation: When wine_type is included in the models, R treats it as a categorical variable rather than a numerical one. This allows the model to recognize “red” and “white” as two different categories and not numerical values.

b) Storage efficiency: Factors are more memory efficient for categorical data compared to character variables. When R recognizes a factor, it stores the variable as integers internally, which makes processing large datasets more efficient.

c) Result Interpretation: Factors allow us to clearly see the influence of each category (red and white in this case) in the models.

Another method for data cleaning is checking for duplicates. Duplicates were found in the dataset however not removed for the following reasons:
Duplicate entries signify repeated observations: In certain instances, duplicates may represent legitimate repeated measurements of the same entity, thereby elucidating the underlying data distribution more effectively. Upon removal, the models may forfeit critical information, leading to diminished performance which was indicated by an increase in RMSE values.
Model Performance and Variability: Certain models, such as Random Forest, exhibit improved performance with more data volume. Guillame-Bert and Teytaud (2018) demonstrated that “Random Forest benefits from being trained on more data, even in the case of already gigantic datasets”. Eliminating duplicates, which reinforce patterns during training, may diminish the influence of specific observations, resulting in increased variability and increase in the number of  mistakes on novel or unseen data. Since the number of observations were less to begin with, removing duplicates led to diminishing the size of the dataset to a great extent which is neither sufficient nor desired for the accuracy of the models. 


