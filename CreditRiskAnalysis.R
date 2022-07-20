# Credit Risk Analysis

### The dataset used on this project is the German Credit Data available on UCI
### https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)



## Uploading the dataset
credit.df <- read.csv("credit_dataset.csv", header = TRUE, sep = ",")


### function for converting to categorical
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

### Function for normalization
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}

### Normalizing the numeric variables
numeric.vars <- c("credit.duration.months", "age", "credit.amount")
credit.df <- scale.features(credit.df, numeric.vars)

### Categorical variables
categorical.vars <- c('credit.rating', 'account.balance', 'previous.credit.payment.status',
                      'credit.purpose', 'savings', 'employment.duration', 'installment.rate',
                      'marital.status', 'guarantor', 'residence.duration', 'current.assets',
                      'other.credits', 'apartment.type', 'bank.credits', 'occupation', 
                      'dependents', 'telephone', 'foreign.worker')

credit.df <- to.factors(df = credit.df, variables = categorical.vars)



### Checking for unbalance on the target variable
library(ggplot2)
ggplot(credit.df,aes_string(x='credit.rating')) + 
  geom_bar()



### Splitting the data on train and test
indexes <- sample(1:nrow(credit.df), size = 0.6 * nrow(credit.df))
train.data <- credit.df[indexes,]
test.data <- credit.df[-indexes,]

## Feature Selection
library(caret) 
library(randomForest) 

### Function to select features
run.feature.selection <- function(num.iters=20, feature.vars, class.var){
  set.seed(10)
  variable.sizes <- 1:10
  control <- rfeControl(functions = rfFuncs, method = "cv", 
                        verbose = FALSE, returnResamp = "all", 
                        number = num.iters)
  results.rfe <- rfe(x = feature.vars, y = class.var, 
                     sizes = variable.sizes, 
                     rfeControl = control)
  return(results.rfe)
}

### Running the feature selection function
rfe.results <- run.feature.selection(feature.vars = train.data[,-1], 
                                     class.var = train.data[,1])


### Feature selection results

varImp((rfe.results))


### ROC curve function

library(ROCR)

plot.roc.curve <- function(predictions, title.text){
  perf <- performance(predictions, "tpr", "fpr")
  plot(perf, col = "black", lty = 1, lwd = 2, 
       main = title.text, cex.main = 0.6, 
       cex.lab = 0.8, xaxs="i", yaxs="i")
  abline(0,1, col = "red")
  auc <- performance(predictions, "auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc,4)
  legend(0.4,0.4, legend = c(paste0("AUC: ",auc)), cex = 0.6, bty = "n", box.col = "white")
  
}

plot.pr.curve <- function(predictions, title.text){
  perf <- performance(predictions, "prec", "rec")
  plot(perf, col = "black", lty = 1, lwd = 2,
       main = title.text, cex.main = 0.6, cex.lab = 0.8, xaxs = "i", yaxs = "i")
}

# FIRST MODEL LOGISTIC REGRESSION


### Creating the first model, without feature selection and balancing the data
library(caret) 
library(ROCR) 


### separate feature and class variables
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]

### Creating a Logistic Regression Model
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
lr.model <- glm(formula = formula.init, data = train.data, family = "binomial")



### Testing the model
lr.predictions <- predict(lr.model, test.data, type="response")
lr.predictions <- round(lr.predictions)

### Evaluating the model
confusionMatrix(table(data = lr.predictions, reference = test.class.var), positive = '1')

### Creating ROC Curve
lr.prediction.values <- predict(lr.model, test.feature.vars, type = "response")
predictions <- prediction(lr.prediction.values, test.class.var)
par(mfrow = c(1,2))
plot.roc.curve(predictions, title.text = "ROC curve First Model: Logistic Regression")
plot.pr.curve(predictions, title.text = "Precision/Recall curve")



# SECOND MODEL: LOGISTIC REGRESSION + FEATURE SELECTION 


### Creating model with Feature selection
formula <- "credit.rating ~ ."
formula <- as.formula(formula)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
model <- train(formula, data = train.data, method = "glm", trControl = control)
importance <- varImp(model, scale = FALSE)
plot(importance)


### Creating the model with the most important variables
formula.new <- "credit.rating ~ account.balance + credit.purpose + previous.credit.payment.status + savings + credit.duration.months"
formula.new <- as.formula(formula.new)
lr.model.new <- glm(formula = formula.new, data = train.data, family = "binomial")



### Testing the model
lr.predictions.new <- predict(lr.model.new, test.data, type = "response") 
lr.predictions.new <- round(lr.predictions.new)

### Evaluating the model
confusionMatrix(table(data = lr.predictions.new, reference = test.class.var), positive = '1')

### Creating ROC Curve
lr.prediction.values <- predict(lr.model.new, test.feature.vars, type = "response")
predictions <- prediction(lr.prediction.values, test.class.var)
par(mfrow = c(1,2))
plot.roc.curve(predictions, title.text = "ROC curve Second Model: Logistic Regression + Feature Selection")
plot.pr.curve(predictions, title.text = "Precision/Recall curve")


# THIRD MODEL: RANDOM FOREST 

library(randomForest)
modelo_rf <- randomForest( credit.rating ~ ., 
                        data = train.data, 
                        ntree = 100, nodesize = 10)



### Testing the model
rf.prediction <- predict(modelo_rf,test.data, type = "response")

### Evaluating the model
confusionMatrix(table(data = rf.prediction, reference = test.class.var), positive = '1')


### Creating ROC Curve
rf.prediction.values <- predict(modelo_rf, test.feature.vars, type = "response")
predictions <- prediction(as.numeric(rf.prediction.values), test.class.var)
par(mfrow = c(1,2))
plot.roc.curve(predictions, title.text = "ROC curve Third Model: Random Forest")
plot.pr.curve(predictions, title.text = "Precision/Recall curve")


# FOURTH MODEL: RANDOM FOREST + FEATURE SELECTION 


library(randomForest)
formula.new <- "credit.rating ~ account.balance + credit.purpose + previous.credit.payment.status + savings + credit.duration.months"
formula.new <- as.formula(formula.new)

modelo_rf_new <- randomForest( formula.new, 
                           data = train.data, 
                           ntree = 100, nodesize = 10)


### Testing the model
rf.prediction_new <- predict(modelo_rf_new,test.data, type = "response")

### Evaluating the model
confusionMatrix(table(data = rf.prediction_new, reference = test.class.var), positive = '1')


### Creating ROC Curve
rf.prediction.values <- predict(modelo_rf_new, test.feature.vars, type = "response")
predictions <- prediction(as.numeric(rf.prediction.values), test.class.var)
par(mfrow = c(1,2))
plot.roc.curve(predictions, title.text = "ROC curve Fourth Model: Random Forest + FS")
plot.pr.curve(predictions, title.text = "Precision/Recall curve")


# FIFTH MODEL: NEURAL NETWORK

library(neuralnet)


data <- read.csv("credit_dataset.csv", header = TRUE, sep = ",")


### Normalize the data
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, 
                              scale = maxs - mins))

### Split the data into training and testing set
index <- sample(1:nrow(data), round(0.6 * nrow(data)))
train_ <- scaled[index,]
test_ <- scaled[-index,]

### Build Neural Network
nn <- neuralnet(credit.rating ~., 
                data = train_, hidden=c(2,1), linear.output=FALSE, threshold=0.01)


previsoes_v1 <- neuralnet::compute(nn, test_[-1])


results <- data.frame(actual = test_[1], prediction = previsoes_v1$net.result)


roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
roundedresultsdf = to.factors(roundedresultsdf, c('credit.rating','prediction'))
caret::confusionMatrix(roundedresultsdf$credit.rating, roundedresultsdf$prediction, positive = '1')

par(mfrow = c(1,1))
library(ROSE)
roc.curve(roundedresultsdf$credit.rating, roundedresultsdf$prediction)


