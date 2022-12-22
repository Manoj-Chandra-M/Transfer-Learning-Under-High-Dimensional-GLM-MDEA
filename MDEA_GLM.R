install.packages("glmtrans", repos = "http://cran.us.r-project.org")
library(glmtrans)
install.packages("readxl")
library("readxl")
D.training <- read_excel(file.choose())
n = 100
target_train <- D.training[row.names(D.training) %in% 1:n, ]
target_source <- D.training[row.names(D.training) %in% (n+1):(6*n), ]
target_test <- D.training[row.names(D.training) %in% (6*n+1):nrow(D.training), ]
target_source_new_1 <- target_source[row.names(target_source) %in% 1:100, ]
target_source_new_2 <- target_source[row.names(target_source) %in% 101:200, ]
target_source_new_3 <- target_source[row.names(target_source) %in% 201:300, ]
target_source_new_4 <- target_source[row.names(target_source) %in% 301:400, ]
target_source_new_5 <- target_source[row.names(target_source) %in% 401:500, ]
train_target_x <- matrix(c(target_train$'Acid gas', target_train$HSAS, target_train$Velocity, target_train$Temperature), nrow=100)
train_target_y <- matrix(c(target_train$'Corrosion Rate'), nrow=100)
target_source_new_1_x <- matrix(c(target_source_new_1$'Acid gas', target_source_new_1$HSAS, target_source_new_1$Velocity, target_source_new_1$Temperature), nrow=100)
target_source_new_1_y <- matrix(c(target_source_new_1$'Corrosion Rate'), nrow=100)
target_source_new_2_x <- matrix(c(target_source_new_2$'Acid gas', target_source_new_2$HSAS, target_source_new_2$Velocity, target_source_new_2$Temperature), nrow=100)
target_source_new_2_y <- matrix(c(target_source_new_2$'Corrosion Rate'), nrow=100)
target_source_new_3_x <- matrix(c(target_source_new_3$'Acid gas', target_source_new_3$HSAS, target_source_new_3$Velocity, target_source_new_3$Temperature), nrow=100)
target_source_new_3_y <- matrix(c(target_source_new_3$'Corrosion Rate'), nrow=100)
target_source_new_4_x <- matrix(c(target_source_new_4$'Acid gas', target_source_new_4$HSAS, target_source_new_4$Velocity, target_source_new_4$Temperature), nrow=100)
target_source_new_4_y <- matrix(c(target_source_new_4$'Corrosion Rate'), nrow=100)
target_source_new_5_x <- matrix(c(target_source_new_5$'Acid gas', target_source_new_5$HSAS, target_source_new_5$Velocity, target_source_new_5$Temperature), nrow=100)
target_source_new_5_y <- matrix(c(target_source_new_5$'Corrosion Rate'), nrow=100)
D.training <- list(target=list(x = train_target_x, y = train_target_y), source=list(list(x1 = target_source_new_1_x, y1 = target_source_new_1_y),list(x2 = target_source_new_2_x, y2 = target_source_new_2_y),list(x3 = target_source_new_3_x, y3 = target_source_new_3_y),list(x4 = target_source_new_4_x, y4 = target_source_new_4_y),list(x5 = target_source_new_5_x, y5 = target_source_new_5_y)))
fit.gaussian <- glmtrans(D.training$target, D.training$source)
test_target_x <- matrix(c(target_test$'Acid gas', target_test$HSAS, target_test$Velocity, target_test$Temperature), nrow=156)
test_target_y <- matrix(c(target_test$'Corrosion Rate'), nrow=156)
D.test <- list(target=list(x = test_target_x, y = test_target_y))
y.pred.glmtrans <- predict(fit.gaussian, D.test$target$x)
mean((y.pred.glmtrans - D.test$target$y)^2)
library(glmnet)
fit.lasso <- cv.glmnet(x = D.training$target$x, y = D.training$target$y)
y.pred.lasso <- predict(fit.lasso, D.test$target$x)
mean((y.pred.lasso - D.test$target$y)^2)


fit.oracle <- glmtrans(target = D.training$target, source = D.training$source, transfer.source.id = 1:2, cores=2)
fit.detection <- glmtrans(target = D.training$target, source = D.training$source, transfer.source.id = "auto", cores=2)
fit.lasso <- cv.glmnet(x = D.training$target$x, y = D.training$target$y)
fit.pooled <- glmtrans(target = D.training$target, source = D.training$source, transfer.source.id = "all", cores=2)
y.pred.oracle <- predict(fit.oracle, D.test$target$x)
y.pred.detection <- predict(fit.detection, D.test$target$x)
y.pred.lasso <- predict(fit.lasso, D.test$target$x)
y.pred.pooled <- predict(fit.pooled, D.test$target$x)
mean((y.pred.oracle - D.test$target$y)^2)
mean((y.pred.detection - D.test$target$y)^2)
mean((y.pred.lasso - D.test$target$y)^2)
mean((y.pred.pooled - D.test$target$y)^2)

install.packages("Metrics", repos = "http://cran.us.r-project.org")
library(Metrics)
mae(D.test$target$y, y.pred.oracle)
mae(D.test$target$y, y.pred.detection)
mae(D.test$target$y, y.pred.lasso)
mae(D.test$target$y, y.pred.pooled)


