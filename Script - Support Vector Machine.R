require(magrittr)
require(e1071)
require(tidyverse)
require(doParallel)
require(foreach)
require(MLmetrics)


df_init <- read_csv("train.csv", col_names = T)


## SVMs don't like NAs
sort(colSums(is.na(df_init)),decreasing = T)

df1 <- df_init %>% 
  mutate(., Age = if_else(is.na(Age), median(Age, na.rm = T), Age),
         Cabin = if_else(is.na(Cabin), "UNKNOWN", Cabin),
         Embarked = if_else(is.na(Embarked), "UNKNOWN", Embarked))


## SibSp and Parch don't tell the whole story about party size
## To fix this, group by ticket number, count, and then join the group back to the data frame
group_ticket <- df1 %>% 
  group_by(., Ticket) %>% 
  count() %>% 
  rename(., party_size = n)

df1 <- left_join(df1, group_ticket, by = c("Ticket"))


## Not sure if the different letters of the cabin mean anything
## So I'm going to separate them and have a look
## Note, when multiple cabins listed, keeping first, dropping rest
df1 %<>% 
  mutate(., Cabin = str_replace_all(Cabin," +[[:alnum:]]+", ""),
         cabin_char = if_else(Cabin == "UNKNOWN", "UNKNOWN", str_replace_all(Cabin, "[[:digit:]]", "")))

## Fare is currently correlated with party_size
## divide by party_size to have Fare stand alone
## Having fare per person should help determine who the wealthier members are (who I assume were more likely to survive)
df1 %<>%
  mutate(., Fare_pp = Fare / party_size)


## Lets bin the ages into age groups
## Generally more thought would go into this, but I'm more just showing off my abilities
## Than I am worried about getting the best score possible
df1 %<>%
  mutate(., age_group = case_when(
    Age < 16 ~"0-16",
    Age < 26 ~"16-26",
    Age < 36 ~"26-36",
    Age < 46 ~ "36-46",
    TRUE ~ "46+"))

# Check groups to make sure they make sense
df1 %>% group_by(., age_group) %>% count() %>% View()


## change qualitative variables to factors
df1 %<>%
  mutate_at(., vars(Survived, Pclass, Sex, Embarked, cabin_char, age_group), as.factor)


## Now let's build the data we'll be training our models on
df2 <- df1 %>% 
  select(., -PassengerId, -Name, -Age, -Ticket, -Fare, -Cabin)

set.seed(123)
train_ind <- sample(seq_len(nrow(df2)), size = 0.7*nrow(df2))

df_train <- df2[train_ind, ]
df_test <- df2[-train_ind, ]



##
###
#### SUPPORT VECTOR MACHINE ####
###
##



## SVM GRID SEARCH 1

# Here I start with a coarse grid
# Will fine tune in later steps
# Can't start fine, because the number of params becomes unwieldy
# Some people like to use a bayesian optimization algorithm
# I used to do that, but I've found that a multi step grid search tends to yield better results
params_svm <- expand.grid(gamma = c(0.001, 0.1, 1, 10), # Gamma defines reach, too small and model can't capture shape of of data,
                                                        # too large and we're in danger of over-fitting
                          cost = c(2^1, 2^5, 2^9, 2^13), # Cost defines the acceptable margin, small means simple function
                                                         # Large means model will try harder to get more values correct
                          epsilon = seq(0.1, 1, 0.2)) # Epsilon works in conjunction with cost, the larger the epsilon,
                                                      # The larger the errors we admit in our solution

# Tuning hyperparameters is embarassingly parallel, so I'll use all 8 of my machines logical processors
registerDoParallel(cores = 8)

svm_opt1 <- foreach(i = 1:nrow(params_svm), .combine = rbind, .packages = "e1071") %dopar% {
  set.seed(123) #this ensures our results are reproducible
  svm(Survived ~.,
      data = df_train,
      type = "C-classification",
      kernel = "radial",
      gamma = params_svm$gamma[i],
      cost = params_svm$cost[i],
      epsilon = params_svm$epsilon[i], 
      cross = 5, # five-fold cross-validation to help ensure results are accurate
      scale = T, # SVM is both faster and more accurate when numeric data is scaled
      probability = T)$tot.accuracy # we select parameters based on test accuracy
  
}
stopImplicitCluster()
svm_opt1 <- data.frame(svm_opt1, params_svm)



## SVM GRID SEARCH 2

params_svm <- expand.grid(gamma = c(0.0001, 0.0005, 0.001, 0.005, 0.01), # Smaller gammas were best
                          cost = c(500, 2000, 5000, 10000, 20000, 50000), # we maxed out cost, so we need to re run with higher limit
                          epsilon = seq(0.1, 1, 0.2)) # Epsilon wasn't a big factor round 1, lets get the first two sorted first

registerDoParallel(cores = 8)

svm_opt2 <- foreach(i = 1:nrow(params_svm), .combine = rbind, .packages = "e1071") %dopar% {
  set.seed(123)
  svm(Survived ~.,
      data = df_train,
      type = "C-classification",
      kernel = "radial",
      gamma = params_svm$gamma[i],
      cost = params_svm$cost[i],
      epsilon = params_svm$epsilon[i], 
      cross = 5,
      scale = T,
      probability = T)$tot.accuracy
  
}
stopImplicitCluster()
svm_opt2 <- data.frame(svm_opt2, params_svm)


## Train SVM model using optimized parameters

set.seed(123)
svm1 <- svm(Survived ~.,
            data = df_train,
            type = "C-classification",
            kernel = "radial",
            gamma = 5e-4,
            cost = 10000,
            epsilon = 0.1, 
            cross = 5,
            scale = T,
            probability = T)

yhat_svm <- predict(svm1, newdata = select(df_test, -Survived), probability = T)
y_svm <- df_test$Survived

svm_sens <- Sensitivity(y_svm, yhat_svm) # True Positive
svm_spec <- Specificity(y_svm, yhat_svm) # True Negative
svm_prec <- Precision(y_svm, yhat_svm) # Total Accuracy


## Cross-Validated Accuracy of 82.0%
## Subsample test data Accuracy of 87.7%







