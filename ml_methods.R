library("data.table")
library("mlr3verse")
library("skimr")
library("dplyr")
library("ggplot2")
skim(data)
set.seed(123) 

data <- read.csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")
head(data)
names(data)

DataExplorer::plot_bar(data, ncol = 4)
DataExplorer::plot_histogram(data, ncol = 4)
DataExplorer::plot_boxplot(data, by = "Personal.Loan", ncol = 4)

min(data$ZIP.Code)
min(data$Experience)

data <- data[data$ZIP.Code>=10000, ] 
data <- data[data$Experience >= 0, ] 

min(data$ZIP.Code)
min(data$Experience)


DataExplorer::plot_boxplot(data, by = "Personal.Loan", ncol = 4)

data$Personal.Loan <- factor(data$Personal.Loan)
train_indices <- sample(1:nrow(data), 0.8 * nrow(data)) 
train <- data[train_indices, ]
test <- data[-train_indices, ]


#Base Line Model
loan_task <- TaskClassif$new(id = "Loan",
                             backend = train, 
                             target = "Personal.Loan"
)

loan_task_test <- TaskClassif$new(id = "Loan test",
                                  backend = test, 
                                  target = "Personal.Loan")


cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(loan_task)

lrn_baseline <- lrn("classif.featureless", predict_type = "prob")

lrn_baseline$train(loan_task)
predictions_base <- lrn_baseline$predict(task = loan_task_test)
predictions_base$confusion
predictions_base$score(list(msr("classif.ce"),
                            msr("classif.acc"),
                            msr("classif.auc"),
                            msr("classif.fpr"),
                            msr("classif.fnr")))

#Trees
dep_vals <- seq(1,25,by=1)
n_vals <- seq(1,100,by=1)

for (i in n_vals) {
  lrn_tree  <- lrn("classif.ranger", predict_type = "prob", num.trees = i, max.depth = 10)
  resample_xgb <- resample(loan_task,lrn_tree,cv5,store_models = TRUE)
  error_n[i] <- resample_xgb$aggregate()
}

for (i in dep_vals){
  lrn_tree  <- lrn("classif.ranger", predict_type = "prob", num.trees = 10, max.depth = i)
  resample_xgb <- resample(loan_task,lrn_tree,cv5,store_models = TRUE)
  error_dep[i] <- resample_xgb$aggregate()
}


require(pspline)
require(KernSmooth)
require(locfit)

plot(n_vals,error_n,xlab = 'Trees', ylab = 'Error')
lp.fit.n <- locpoly(n_vals,error_n,bandwidth = 5)
lines(lp.fit.n,col=3,lwd=3)


plot(dep_vals,error_dep,xlab = 'Depth',ylab = 'Error')
lp.fit.dep <- locpoly(dep_vals,error_dep,bandwidth = 2)
lines(lp.fit.dep,col=3,lwd=3)


lrn_tree <- lrn("classif.ranger", predict_type = "prob")

lrn_tree$train(loan_task)
predictions_tree <- lrn_tree$predict(task = loan_task_test)
predictions_tree$confusion
predictions_tree$score(list(msr("classif.ce"),
                            msr("classif.acc"),
                            msr("classif.auc"),
                            msr("classif.fpr"),
                            msr("classif.fnr")))





lrn_tree <- lrn("classif.ranger", predict_type = "prob", max.depth = 10, num.trees = 20)

lrn_tree$train(loan_task)
predictions_tree <- lrn_tree$predict(task = loan_task_test)
predictions_tree$confusion
predictions_tree$score(list(msr("classif.ce"),
                            msr("classif.acc"),
                            msr("classif.auc"),
                            msr("classif.fpr"),
                            msr("classif.fnr")))

#log reg

lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")
lrn_log_reg$train(loan_task)
predictions_log_reg <- lrn_log_reg$predict(task = loan_task_test)
predictions_log_reg$confusion
predictions_log_reg$score(list(msr("classif.ce"),
                               msr("classif.acc"),
                               msr("classif.auc"),
                               msr("classif.fpr"),
                               msr("classif.fnr")))

#xgboost

n_vals <- seq(1,100,by=1)
dep_vals <- seq(1,25,by=1)

error_n_boost <- matrix(NA, nrow = length(n_vals))
error_dep_boost <- matrix(NA, nrow = length(dep_vals))

for (i in n_vals) {
  lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob", nrounds = i, max_depth = 5)
  resample_xgb <- resample(loan_task,lrn_xgboost,cv5,store_models = TRUE)
  error_n_boost[i] <- resample_xgb$aggregate()
}

for (i in dep_vals){
  lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob", nrounds = 10, max_depth = i)
  resample_xgb <- resample(loan_task,lrn_xgboost,cv5,store_models = TRUE)
  error_dep_boost[i] <- resample_xgb$aggregate()
}

plot(n_vals,error_n_boost,xlab = 'Number of Rounds',ylab = 'Error')
lp.fit.n <- locpoly(n_vals,error_n_boost,bandwidth = 7)
lines(lp.fit.n,col=2,lwd=3)


plot(dep_vals,error_dep_boost,xlab = 'Max Depth',ylab = 'Error')
lp.fit.dep <- locpoly(dep_vals,error_dep_boost,bandwidth = 1)
lines(lp.fit.dep,col=2,lwd=3)



lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_xgboost$train(loan_task)
predictions_xgboost <- lrn_xgboost$predict(task = loan_task_test)
predictions_xgboost$confusion
predictions_xgboost$score(list(msr("classif.ce"),
                               msr("classif.acc"),
                               msr("classif.auc"),
                               msr("classif.fpr"),
                               msr("classif.fnr")))



lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob", nrounds = 50, max_depth = 5)
lrn_xgboost$train(loan_task)
predictions_xgboost <- lrn_xgboost$predict(task = loan_task_test)
predictions_xgboost$confusion
predictions_xgboost$score(list(msr("classif.ce"),
                               msr("classif.acc"),
                               msr("classif.auc"),
                               msr("classif.fpr"),
                               msr("classif.fnr")))

#Neural Network
library("rsample")

data_split <- initial_split(data,prop = 80/100)
data_train <- training(data_split)
data_split2 <- initial_split(testing(data_split), 0.5)
data_validate <- training(data_split2)
data_test <- testing(data_split2)

library("recipes")
cake <- recipe(Personal.Loan ~ ., data = data) %>%
  step_impute_mean(all_numeric()) %>% 
  step_center(all_numeric()) %>% 
  step_scale(all_numeric()) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  prep(training = data_train) 

data_train_final <- bake(cake, new_data = data_train)
data_validate_final <- bake(cake, new_data = data_validate)  
data_test_final <- bake(cake, new_data = data_test)

library("tensorflow")
library("keras")


data$Family <- factor(data$Family)
data$Education <- factor(data$Education)
data$Securities.Account <- factor(data$Securities.Account)
data$Online <- factor(data$Online)
data$CD.Account <- factor(data$CD.Account)
data$CreditCard <- factor(data$CreditCard)


#X1
data_train_x <- data_train_final %>%
  select(-starts_with("Personal.Loan_")) %>%
  as.matrix()
data_train_y <- data_train_final %>%
  select(Personal.Loan_X1) %>%
  as.matrix()

data_validate_x <- data_validate_final %>%
  select(-starts_with("Personal.Loan_")) %>%
  as.matrix()
data_validate_y <- data_validate_final %>%
  select(Personal.Loan_X1) %>%
  as.matrix()

data_test_x <- data_test_final %>%
  select(-starts_with("Personal.Loan_")) %>%
  as.matrix()
data_test_y <- data_test_final %>%
  select(Personal.Loan_X1) %>%
  as.matrix()

deep.net <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "sigmoid",
              input_shape = c(ncol(data_train_x))) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 32, activation = "sigmoid") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 32, activation = "sigmoid") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 32, activation = "sigmoid") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")


deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

deep.net %>% fit(
  data_train_x, data_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(data_validate_x, data_validate_y),
)

pred_test_prob <- deep.net %>% predict(data_test_x)
pred_test_res <- deep.net %>% predict(data_test_x) %>% `>`(0.5) %>% as.integer()
table(pred_test_res, data_test_y)
yardstick::accuracy_vec(as.factor(data_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(factor(data_test_y, levels = c("1","0")),
                       c(pred_test_prob))


classification_error <- mean(pred_test_res != data_test_y)
classification_error

conf_matrix <- table(pred_test_res, data_test_y)


TN <- conf_matrix[1, 1]
FP <- conf_matrix[1, 2]
FN <- conf_matrix[2, 1]
TP <- conf_matrix[2, 2]

FPR <- FP / (FP + TN)
FNR <- FN / (FN + TP)
print(paste("False Positive Rate (FPR):", FPR))
print(paste("False Negative Rate (FNR):", FNR))


#roc curve

library(pROC)






p_base <- predictions_base$prob[,2]
p_tree <- predictions_tree$prob[,2]
p_log_reg <- predictions_log_reg$prob[,2]
p_boost <- predictions_xgboost$prob[,2]
p_nn <- pred_test_prob
roc_curve_base <- roc(test$Personal.Loan, p_base)
roc_curve_tree <- roc(test$Personal.Loan, p_tree)
roc_curve_log_reg <- roc(test$Personal.Loan, p_log_reg)
roc_curve_boost <- roc(test$Personal.Loan, p_boost)
roc_curve_nn <- roc(data_test_y, pred_test_prob)

roc_base <- coords(roc_curve_base)
xbase <- roc_base$specificity
ybase <- roc_base$sensitivity

roc_tree <- coords(roc_curve_tree)
xtree <- roc_tree$specificity
ytree <- roc_tree$sensitivity


roc_log_reg <- coords(roc_curve_log_reg)
xlog_reg <- roc_log_reg$specificity
ylog_reg <- roc_log_reg$sensitivity

roc_boost <- coords(roc_curve_boost)
xboost <- roc_boost$specificity
yboost <- roc_boost$sensitivity

roc_nn <- coords(roc_curve_nn)
xnn <- roc_nn$specificity
ynn <- roc_nn$sensitivity

plot(1-xbase,ybase,
     main = 'ROC',
     xlab = 'FPR',
     ylab = 'TPR',
     col = 'black',
     type = 'l',
     xlim = c(0,1),
     ylim = c(0,1),lwd=2)
lines(1-xtree,ytree, col='green',lwd=2)
lines(1-xlog_reg,ylog_reg, col = 'blue',lwd=2)
lines(1-xboost,yboost, col = 'red',lwd=2)
lines(1-xnn,ynn, col = 'purple',lwd=2)
legend("bottomright", 
       legend = c("Base", "Tree", "Logistic Regression", "Gradient Boosting", "Neural Network"),
       col = c("black", "green", "blue", "red", "purple"),
       lty = 1,
       cex = 0.8)

#precall
library(PRROC)
#https://cran.r-project.org/web/packages/PRROC/vignettes/PRROC.pdf

weights <- as.numeric(as.character(test$Personal.Loan))
weight_nn <- as.numeric(as.character(data_test_y))

pr_tree <-pr.curve(scores.class0 = p_tree, weights.class0 = weights, curve=TRUE)
pr_log_reg <- pr.curve(scores.class0 = p_log_reg, weights.class0 = weights, curve=TRUE)
pr_boost<-pr.curve(scores.class0 = p_boost, weights.class0 = weights, curve=TRUE)
pr_nn <- pr.curve(scores.class0 = p_nn, weights.class0 = weight_nn, curve=TRUE)

plot(pr_tree, col = "green", main = "Precision-Recall Curves", xlab = "Recall", ylab = "Precision",auc.main = FALSE,lwd=2)


plot(pr_log_reg, col = "blue",add = TRUE,lwd=2)
plot(pr_boost, col = "red",add = TRUE,lwd=2)
plot(pr_nn, col = "purple",add = TRUE,lwd=2)
legend("bottomleft", 
       legend = c("Tree", "Logistic Regression", "Gradient Boosting", "Neural Network"),
       col = c( "green", "blue", "red", "purple"),
       lty = 1,
       cex = 0.8)

#calibration curve
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob",nrounds = 20, max_depth = 5)
lrn_xgboost$train(loan_task)
predictions_xgboost <- lrn_xgboost$predict(task = loan_task_test)
predictions_xgboost$confusion
predictions_xgboost$score(list(msr("classif.ce"),
                               msr("classif.acc"),
                               msr("classif.auc"),
                               msr("classif.fpr"),
                               msr("classif.fnr")))





pred_probs <- predictions_xgboost$prob[, 2]  
p_tree <- predictions_tree$prob[,2]
p_log_reg <- predictions_log_reg$prob[,2]
p_nn <- pred_test_prob

true_labels <- test$Personal.Loan

calibration_data_xgboost <- data.frame("Predicted_Prob" = pred_probs, "True_Label" = true_labels)
calibration_data_tree <- data.frame("Predicted_Prob" = p_tree, "True_Label" = true_labels)
calibration_data_log <- data.frame("Predicted_Prob" = p_log_reg, "True_Label" = true_labels)
calibration_data_nn <- data.frame("Predicted_Prob" = p_nn, "True_Label" = data_test_y)

num_bins <- 10
bin_width <- 1 / num_bins


observed_event_percentages_xgboost <- numeric(num_bins)
observed_event_percentages_tree <- numeric(num_bins)
observed_event_percentages_log <- numeric(num_bins)
observed_event_percentages_nn <- numeric(num_bins)
bin_midpoints <- numeric(num_bins)

for (i in 1:num_bins) {
  lower_bound <- (i - 1) * bin_width
  upper_bound <- i * bin_width
  
  
  bin_data_xgboost <- calibration_data_xgboost[calibration_data_xgboost$Predicted_Prob >= lower_bound & calibration_data_xgboost$Predicted_Prob < upper_bound, ]
  observed_event_percentages_xgboost[i] <- sum(bin_data_xgboost$True_Label==1) / nrow(bin_data_xgboost)
  
  
  bin_data_tree <- calibration_data_tree[calibration_data_tree$Predicted_Prob >= lower_bound & calibration_data_tree$Predicted_Prob < upper_bound, ]
  observed_event_percentages_tree[i] <- sum(bin_data_tree$True_Label==1) / nrow(bin_data_tree)
  
  bin_data_log <- calibration_data_log[calibration_data_log$Predicted_Prob >= lower_bound & calibration_data_log$Predicted_Prob < upper_bound, ]
  observed_event_percentages_log[i] <- sum(bin_data_log$True_Label==1) / nrow(bin_data_log)
  
  bin_data_nn <- calibration_data_nn[calibration_data_nn$Predicted_Prob >= lower_bound & calibration_data_nn$Predicted_Prob < upper_bound, ]
  observed_event_percentages_nn[i] <- sum(bin_data_nn$True_Label==1) / nrow(bin_data_nn)
  
  bin_midpoints[i] <- (lower_bound + upper_bound) / 2
}

plot(bin_midpoints, observed_event_percentages_xgboost, type = "o", 
     xlab = "Predicted Probability/Bin midpoint", ylab = "Observed Event Percentage",
     main = "Calibration Curve",
     col = 'red')
lines(bin_midpoints, observed_event_percentages_tree,type = 'o' ,col = 'green')
lines(bin_midpoints, observed_event_percentages_log,type = 'o' ,col = 'blue')
#lines(bin_midpoints, observed_event_percentages_nn,type = 'o' ,col = 'purple')
abline(0, 1, col = "grey")

legend("bottomright", 
       legend = c("XGBoost", "Tree", "Logistic Regression"),
       col = c("red", "green", "blue"),
       lty = 1,
       cex = 0.8,
       title = "Models")





