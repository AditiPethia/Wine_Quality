# Install necessary libraries
# Load necessary libraries
library(dplyr)
library(ggplot2)
library(corrplot)
library(glmnet)
library(randomForest)
library(caret)
library(reshape2)

# Function to calculate RMSE
calculate_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Function to calculate R²
calculate_r2 <- function(actual, predicted) {
  cor(actual, predicted)^2
}

# Load datasets
red_wine <- read.csv("winequality-red.csv")     #add file path
white_wine <- read.csv("winequality-white.csv")   #add file path

# Add wine type
red_wine$wine_type <- "red"
white_wine$wine_type <- "white"

# Combine and convert
wine_data <- bind_rows(red_wine, white_wine)
wine_data$wine_type <- as.factor(wine_data$wine_type)
str(wine_data)


# Compute correlation of all variables with 'quality'
numeric_vars <- wine_data[, sapply(wine_data, is.numeric)]
cor_values <- cor(numeric_vars, use = "complete.obs")

# Plot correlation matrix using corrplot
corrplot(cor_values, method = "color", type = "full",
         addCoef.col = "black", tl.col = "black", 
         tl.cex = 0.8, number.cex = 0.7, diag = FALSE)

# ----Distribution of variables ----
for (var in names(wine_data)) {
  if (is.numeric(wine_data[[var]])) {
    p <- ggplot(wine_data, aes(x = .data[[var]])) +
      geom_histogram(fill = "skyblue", color = "black", bins = 40) +
      labs(title = paste("Distribution of", var), x = var, y = "Count") +
      theme_minimal()
    print(p)
  }
}

# ----Box plot before winsorization ----
par(mfrow = c(2,2))
for (var in names(wine_data)) {
  if (is.numeric(wine_data[[var]])) {
    boxplot(wine_data[[var]], main = paste("Boxplot (Before)", var),
            col = "lightblue", border = "black", horizontal = TRUE)
  }
}
par(mfrow = c(1,1))

# Detect and count outliers
detect_outliers <- function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR)
}
numeric_cols <- sapply(wine_data, is.numeric)
outliers <- sapply(wine_data[, numeric_cols], detect_outliers)
colSums(outliers)

# Winsorization
winsorize <- function(x) {
  lower <- quantile(x, 0.01)
  upper <- quantile(x, 0.99)
  x[x < lower] <- lower
  x[x > upper] <- upper
  return(x)
}
wine_data_winsorized <- wine_data
wine_data_winsorized[, !(names(wine_data) %in% c("wine_type", "quality"))] <- 
  lapply(wine_data_winsorized[, !(names(wine_data) %in% c("wine_type", "quality"))], winsorize)

# ----Box plots after winsorization ----
par(mfrow = c(2,2))
for (var in names(wine_data_winsorized)) {
  if (is.numeric(wine_data_winsorized[[var]])) {
    boxplot(wine_data_winsorized[[var]], main = paste("Boxplot (After)", var),
            col = "lightgreen", border = "darkgreen", horizontal = TRUE)
  }
}
par(mfrow = c(1,1))

# Log transformation
wine_data_log <- wine_data
wine_data_log[, !(names(wine_data) %in% c("wine_type", "quality"))] <- 
  lapply(wine_data_log[, !(names(wine_data) %in% c("wine_type", "quality"))], function(x) log1p(x))

# Stratified train-test split
set.seed(123)
trainIndex <- createDataPartition(wine_data$wine_type, p = 0.8, list = FALSE)
y <- wine_data$quality
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# ---- Lasso regression ----
X <- model.matrix(quality ~ ., data = wine_data)[, -1]
X_train <- X[trainIndex, ]; X_test <- X[-trainIndex, ]
lasso_model <- cv.glmnet(X_train, y_train, alpha = 1)
predictions <- predict(lasso_model, X_test, s = lasso_model$lambda.min)
baseline_rmse <- calculate_rmse(predictions, y_test)
lasso_r2 <- calculate_r2(y_test, as.vector(predictions))

X_winsorized <- model.matrix(quality ~ ., data = wine_data_winsorized)[, -1]
lasso_model_winsorized <- cv.glmnet(X_winsorized[trainIndex, ], y_train, alpha = 1)
pred_winsorized <- predict(lasso_model_winsorized, X_winsorized[-trainIndex, ], s = "lambda.min")
winsorized_rmse <- calculate_rmse(pred_winsorized, y_test)
lasso_r2_winsorized <- calculate_r2(y_test, as.vector(pred_winsorized))

X_log <- model.matrix(quality ~ ., data = wine_data_log)[, -1]
lasso_model_log <- cv.glmnet(X_log[trainIndex, ], y_train, alpha = 1)
pred_log <- predict(lasso_model_log, X_log[-trainIndex, ], s = "lambda.min")
log_rmse <- calculate_rmse(pred_log, y_test)
lasso_r2_log <- calculate_r2(y_test, as.vector(pred_log))

# Compare RMSE for each model (Lasso)
lasso_results <- data.frame(
  Method = c("Original", "Winsorization", "Log Transformation"),
  RMSE = c(baseline_rmse, winsorized_rmse, log_rmse)
)

print("Lasso Model RMSE Comparison:")
print(lasso_results)


# ---- Random Forest ----
rf_model <- randomForest(quality ~ ., data = wine_data, subset = trainIndex, ntree = 500)
rf_pred <- predict(rf_model, wine_data[-trainIndex, ])
rf_rmse <- calculate_rmse(rf_pred, y_test)
rf_r2 <- calculate_r2(y_test, rf_pred)

rf_model_winsorized <- randomForest(quality ~ ., data = wine_data_winsorized, subset = trainIndex, ntree = 500)
rf_pred_winsorized <- predict(rf_model_winsorized, wine_data_winsorized[-trainIndex, ])
rf_rmse_winsorized <- calculate_rmse(rf_pred_winsorized, y_test)
rf_r2_winsorized <- calculate_r2(y_test, rf_pred_winsorized)

rf_model_log <- randomForest(quality ~ ., data = wine_data_log, subset = trainIndex, ntree = 500)
rf_pred_log <- predict(rf_model_log, wine_data_log[-trainIndex, ])
rf_rmse_log <- calculate_rmse(rf_pred_log, y_test)
rf_r2_log <- calculate_r2(y_test, rf_pred_log)

# Compare RMSE for Random Forest model
rf_results <- data.frame(
  Method = c("Original", "Winsorization", "Log Transformation"),
  RMSE = c(rf_rmse, rf_rmse_winsorized, rf_rmse_log)
)

print("Random Forest Model RMSE Comparison:")
print(rf_results)


# ---- Linear regression ----
lm_model <- lm(quality ~ ., data = wine_data, subset = trainIndex)
lm_pred <- predict(lm_model, newdata = wine_data[-trainIndex, ])
lm_rmse <- calculate_rmse(lm_pred, y_test)
lm_r2 <- calculate_r2(y_test, lm_pred)

lm_model_winsorized <- lm(quality ~ ., data = wine_data_winsorized, subset = trainIndex)
lm_pred_winsorized <- predict(lm_model_winsorized, wine_data_winsorized[-trainIndex, ])
lm_rmse_winsorized <- calculate_rmse(lm_pred_winsorized, y_test)
lm_r2_winsorized <- calculate_r2(y_test, lm_pred_winsorized)

lm_model_log <- lm(quality ~ ., data = wine_data_log, subset = trainIndex)
lm_pred_log <- predict(lm_model_log, wine_data_log[-trainIndex, ])
lm_rmse_log <- calculate_rmse(lm_pred_log, y_test)
lm_r2_log <- calculate_r2(y_test, lm_pred_log)

# Compare RMSE for linear regression model
lm_results <- data.frame(
  Method = c("Original", "Winsorization", "Log Transformation"),
  RMSE = c(lm_rmse, lm_rmse_winsorized, lm_rmse_log)
)

print("Liner regession Model RMSE Comparison:")
print(lm_results)


# ---- Final comparison table ----
final_model_comparison <- data.frame(
  Model = c("Lasso", "Random Forest", "Linear Regression"),
  Transformation = c("Winsorized", "Log Transformed", "Winsorized"),
  RMSE = c(winsorized_rmse, rf_rmse_log, lm_rmse_winsorized),
  R_Squared = c(lasso_r2_winsorized, rf_r2_log, lm_r2_winsorized)
)
print("Best Models with Transformations:")
print(final_model_comparison)


# ---- Feature importance ----
# Lasso Regression
lasso_coeff <- coef(lasso_model_winsorized, s = "lambda.min")
lasso_imp <- data.frame(
  Feature = rownames(lasso_coeff)[-1],
  Importance = abs(as.vector(lasso_coeff[-1]))
)
lasso_imp <- lasso_imp[order(-lasso_imp$Importance), ]

# Random Forest
rf_imp <- importance(rf_model_log)
rf_imp_df <- data.frame(
  Feature = rownames(rf_imp),
  Importance = rf_imp[, 1]
)
rf_imp_df <- rf_imp_df[order(-rf_imp_df$Importance), ]

# Linear Regression
lm_imp_df <- data.frame(
  Feature = names(lm_model_winsorized$coefficients)[-1],
  Importance = abs(lm_model_winsorized$coefficients[-1])
)
lm_imp_df <- lm_imp_df[order(-lm_imp_df$Importance), ]

# ---- Visualizations ----
# RMSE and R² plot
comparison_long <- melt(final_model_comparison, id.vars = c("Model", "Transformation"))
ggplot(comparison_long, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(value, 2)), vjust = -0.5, position = position_dodge(0.9), size = 3.5) +
  facet_wrap(~variable, scales = "free_y") +
  labs(title = "Final Model Comparison (RMSE and R²)",
       x = "Model", y = "Value") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Tilt x-axis labels for readibility

# Feature importance plots
top_n <- 10
ggplot(head(lasso_imp, top_n), aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "skyblue") + coord_flip() +
  labs(title = "Lasso (Winsorized) Feature Importance") + theme_minimal()

ggplot(head(rf_imp_df, top_n), aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "lightgreen") + coord_flip() +
  labs(title = "Random Forest (Log) Feature Importance") + theme_minimal()

ggplot(head(lm_imp_df, top_n), aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "coral") + coord_flip() +
  labs(title = "Linear Regression (Winsorized) Feature Importance") + theme_minimal()




