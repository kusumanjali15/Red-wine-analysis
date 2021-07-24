# Red-wine-analysis

Here we take Red wine dataset and perform analysis by using machine learning algorithms like Linear Regression, Ridge Regression and Lasso Regression.
By applying these algorithms we find Mean Squared Error, R Squared and Adjusted R Squared values and then compare the algorithms and find which algorithm is better algorithm.


View(wineQualityReds)

install.packages("caret")
install.packages("glmnet")
install.packages("mlbench")
install.packages("party")
install.packages("psych")
library(caret)
library(glmnet)
library(mlbench)
library(psych)
library(party)


head(wineQualityReds)
summary(wineQualityReds)

#correlation matrix
cor(wineQualityReds)

#scatter plot matrix
pairs(wineQualityReds[,5:12])

#boxplot
boxplot(wineQualityReds$fixed.acidity,wineQualityReds$volatile.acidity,wineQualityReds$citric.acid,wineQualityReds$residual.sugar,wineQualityReds$chlorides,wineQualityReds$density)
boxplot(chlorides~quality,data = wineQualityReds)
boxplot(wineQualityReds)

#linear regression
linear_model = lm(formula = quality~.,data = wineQualityReds)
summary(linear_model)
lm(formula = quality~chlorides*sulphates,data = wineQualityReds)

#splitting data
set.seed(1)
ind = sample(2,nrows,replace = T,prob = c(0.7,0.3))
train = wineQualityReds[ind==1,]
test = wineQualityReds[ind==2,]

#custom control parameters
custom = trainControl(method = "repeatedcv",number = 10,repeats = 5,verboseIter = T)
custom

#ridge regression
set.seed(1)
ridge=train(quality~.,wineQualityReds,method = 'glmnet',
            tuneGrid=expand.grid(alpha=0,lambda=seq(0.00001,5,length=5)),trControl=custom)

#plot results
plot(ridge)
ridge
plot(ridge$finalModel,xvar = "lambda",label = T)
plot(ridge$finalModel,xvar="dev",label=T)
plot(varImp(ridge,scale = T))

best=ridge$finalModel
coef(best,s=ridge$bestTune$lambda)

#lasso regression
set.seed(1)
lasso=train(quality~.,wineQualityReds,method = 'glmnet',
            tuneGrid=expand.grid(alpha=1,lambda=seq(0.00001,5,length=5)),trControl=custom)

#plot results
plot(lasso)
lasso
plot(lasso$finalModel,xvar = "lambda",label = T)
plot(lasso$finalModel,xvar="dev",label=T)
plot(varImp(ridge,scale = T))

best=lasso$finalModel
coef(best,s=lasso$bestTune$lambda)

#compare the models
model_list=list(Ridge=ridge,Lasso=lasso)
results=resamples(model_list)
summary(results)

