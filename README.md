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

a) Duplicate entries signify repeated observations: In certain instances, duplicates may represent legitimate repeated measurements of the same entity, thereby elucidating the underlying data distribution more effectively. Upon removal, the models may forfeit critical information, leading to diminished performance which was indicated by an increase in RMSE values.

b)Model Performance and Variability: Certain models, such as Random Forest, exhibit improved performance with more data volume. Guillame-Bert and Teytaud (2018) demonstrated that “Random Forest benefits from being trained on more data, even in the case of already gigantic datasets”. Eliminating duplicates, which reinforce patterns during training, may diminish the influence of specific observations, resulting in increased variability and increase in the number of  mistakes on novel or unseen data. Since the number of observations were less to begin with, removing duplicates led to diminishing the size of the dataset to a great extent which is neither sufficient nor desired for the accuracy of the models. 

# Exploratory Data Analysis
## Distribution of Predictor Variables

The Exploratory Data Analysis began with checking the distribution of each numerical predictor variable. Histograms were created for each variable, to graphically illustrate their distribution, in order to highlight significant attributes of the dataset.  Multiple predictors showed significant skewness:
The residual sugar showed a prominent positive skew, with the majority of  values clustered at the lower end of the spectrum and a lengthy tail extending towards elevated values.
Chlorides showed a positive skew, with values concentrated at the lower end.
Both free sulphur dioxide and total sulphur dioxide demonstrated right-skewed distributions.
Volatile acidity and citric acid showed a more modest skewness.
The distribution patterns suggested that data transformations would be helpful and beneficial prior to modelling to rectify the non-normal distributions of numerous predictors.

## Correlation of Predictor Variables with Target Variable

To see which variable(s) impacted the quality of wine the most, correlation analysis was performed and matrix was plotted for visualization.
From the heat map above the following observations can be made:
Sure! Here's the same information in **bullet point format** instead of a table:

---

### 🔍 Observations from the Heat Map:

- **Alcohol (Correlation: 0.44)**  
  → Strongest positive correlation with wine quality.  
  → Higher alcohol content enhances complexity and body, leading to better ratings.

- **Volatile Acidity (Correlation: -0.27)**  
  → Moderate negative correlation.  
  → Higher acidity tends to lower wine quality.

- **Sulphates (Correlation: 0.19)**  
  → Weak positive correlation.  
  → Acts as a preservative and antioxidant, slightly improving quality scores.

- **Density (Correlation: -0.31)**  
  → Moderate negative correlation.  
  → Higher density (often from residual sugar) is associated with lower quality.

- **Citric Acid (Correlation: 0.09)**  
  → Weak positive correlation.  
  → Acts as a preservative, but with limited effect on quality.

- **Total Sulfur Dioxide (Correlation: -0.04)**  
  → Very weak negative correlation.  
  → While it's used to preserve wine, high levels may reduce perceived quality.

- **Chlorides (Correlation: -0.04)**  
  → Very weak negative correlation.  
  → Indicates that salty wines may have slightly lower quality.

- **Fixed Acidity (Correlation: -0.08)**  
  → Very weak negative correlation.  
  → Suggests that fixed acidity alone doesn’t strongly influence wine quality.

- **pH (Correlation: -0.31)**  
  → Moderate negative correlation.  
  → Lower pH (more acidic) wines are generally rated higher for their freshness.

- **Free Sulfur Dioxide (Correlation: 0.02)**  
  → Very weak positive correlation.  
  → Helps prevent oxidation but has minimal effect on quality ratings.

- **Residual Sugar (Correlation: -0.04)**  
  → Very weak negative correlation.  
  → Sweeter wines aren’t necessarily rated higher; sweetness isn’t a major quality factor.

## Target Variable Distribution
Graph of target variable (quality) indicated a nearly normal distribution centred on the median quality scores, albeit with notable peculiarities.
Quality scores lie between 3-9 on a scale 0f 0-10 and majority of the wines obtained ratings between 5 and 6, resulting in a central peak. Another point to be noted was that a very small fraction of wines got a score on the extremes either 0-2 or 8-9, thereby suggesting an average selection of wine. The distribution exhibited a minor skew towards elevated ratings.The class imbalance is a critical factor in model training and assessment.

## Handling Outliers
Outliers were detected using IQR method and visualized using boxplots. Outliers often create bias in data analysis which is why it is necessary to handle them before further analysis.
Winsorization was applied to cap extreme values at the 1st and 99th quartile. This retains the overall distribution of the data while still minimizing the effect of outliers on the analysis. Outliers were not removed since they were naturally occurring variations in observation and not errors. Moreover removing values would reduce the size of the dataset. Removing outliers entirely could have led to a loss of valuable information, especially in a dataset where extreme values might still contain meaningful variations in wine characteristics, such as differences in alcohol content or acidity that could still influence wine quality. This method also ensures that the model is not overly sensitive to any single data point, resulting in more stable predictions.

## Log Transformation
“Log transformation is a data transformation method in which it replaces each variable x with a log(x)”(Htut, 2020). The idea is to change the original data, which may or may not follow a normal distribution, into one that does. When data is normally distributed, it shows linear relation between the dependent and independent variables, hence to establish this relation and to enhance the linearity, log transformations are applied to the data. It reduces skewness of variables thereby making them suitable for models that assume a normally distributed input.

## Train- Test Split
To evaluate the models’ generalisation ability, the data was split into a training set (80%) and test set (20%). This was done to ensure that some part of the data remains unseen by the  model that could  lead to the model simply memorizing the dataset. The data was split by wine_type and not quality variable. This was done keeping in mind the difference in number of observations for white and red wine. Without the 80:20 split by wine type, the model was biased towards white wine samples only since there was more data available for it. The model naturally classified red wine of poor quality due to the sheer lack of observations. Using 80% training data would allow the model to work with relatively equal amounts of red and white data.
Furthermore, stratified sampling was used to maintain distribution of quality scores in training and testing sets. This ensures that both the training and test sets are representative of the full range of wine quality scores, preventing any bias that could arise from an imbalanced distribution of quality scores. 
In order to avoid overfitting or underfitting of data, the split was done on the basis of the wine_type variable to account for class imbalance between red and white wines and also to accommodate the differences in chemical composition of the two types. Moreover, by splitting the data by wine type it was ensured that the models learn patterns specific to each wine type. 

# Machine Learning Models

## LASSO Regression
LASSO stands for Least Absolute Shrinkage and Selection Operator. This particular model “shrinks” or reduces the regression coefficients to zero in order to prevent overfitting of data. It gives a poor value with training set but gives more stable and reliable result switch unseen data or test sets. Shrinking certain coefficients to zero also helps in feature selection. The key advantage LASSO Regression has over traditional Linear Regression is its ability to handle data with several variables by reducing the complexity of the model, making it more interpretable. Lasso was selected for its ability to not only predict wine quality but also identify which chemical properties are the most influential.

## Random Forest
Random forest, as the name suggests, comprises random decision trees that work alongside classification and regression techniques(B. Shaw et al, 2020). Each tree works on a random subset of the data, and the final outcome is computed using averages from all the decision trees. This approach reduces variance and enhances stability of the model.
Random forest can be particularly useful in cases of non linearity between predictor variables and target variables. A notable fact is that wine quality may be a result of complex reactions and interactions of multiple features.  For this reason the random forest model posed to be beneficial as it can model these interactions. 
In this study, Random Forest was trained with 500 decision trees. The feature importance was calculated using the mean decrease in impurity, which indicates how much each feature contributes to the reduction in variance across all trees.

## Linear Regression
Linear regression was used as a baseline model for the more complex models. Even though linear regression assumes linear relation between predictor variable and target variable, the results are easy to interpret and can act as a mere starting point. That being said, linear regression is not as inaccurate as the more complex models LASSO or random forest or any other model. 

## Model Comparison
All three models, namely, LASSO Regression, Linear Regression and Random Forest model were run on three sets of data, original, winsorized and log transformed. This comparison of RMSE helped in establishing which data is best suited for which model respectively. Lower the RMSE better is the accuracy of a model. 

It was seen that LASSO and Random first benefit from different transformations of data and overall, th eRandom forest model is the most accurate with the minimum RMSE value and the highest R2 value. 
The LASSO model performed best with winsorized data since it is highly sensitive to extreme outliers. Winsorization caps the extreme values, thereby diminishing their effect on the model. Log transformation however, only changes the distribution of data without affecting the outliers. LASSO might still get affected by the largely deviated values. 
On the contrary, the Random Forest model is robust and not sensitive to outliers. However it is sensitive to the distribution of data. Hence, log transformed data helped improve the performance of the Random forest model. From this comparison it is clear that the Random Forest model is the most accurate due to lower RMSE and higher R2.

# Conclusion

The results clearly indicate that Lasso Regression with Winsorized data and Random Forest with log-transformed data each performed optimally with their respective preprocessing techniques. The Winsorization in Lasso Regression helped mitigate the effect of extreme values, leading to better model stability and performance. For Random Forest, the log transformation helped balance skewed data distributions, which allowed the model to make better splits and capture complex non-linear relationships, resulting in superior predictive accuracy.
While both Lasso and Random Forest showed improvements with their respective transformations, the key takeaway is the importance of selecting the right preprocessing technique for each model. Lasso benefits from the reduction of outlier influence through Winsorization, which stabilizes the model, while Random Forest thrives on the normalization of input data via log transformation, which improves its ability to learn from complex interactions in the data.
The performance improvements seen with both models emphasize the importance of data preprocessing in enhancing the accuracy and reliability of machine learning models, particularly when dealing with datasets containing skewed distributions or extreme values.















