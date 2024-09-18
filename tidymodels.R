library(tidymodels)
library(GGally)
library(glmnet)
library(tidyverse)

heart_failure <- readr::read_csv("data/heart_failure.csv") 

heart_failure <- heart_failure |> 
  select(age, sex, smoking, anaemia, diabetes, high_blood_pressure,
         serum_creatinine, serum_creatinine, creatinine_phosphokinase,
         platelets, ejection_fraction, time, DEATH_EVENT) |> 
  rename(death = DEATH_EVENT) |> 
         sex = case_when(sex == 0 ~ "F",
                         sex == 1 ~ "M") |> 
  mutate_at(c("death", "smoking", "anaemia", "diabetes", "high_blood_pressure"), as.factor)


# Inspect variables -------------------------------------------------------

barplot(table(heart_failure$death))
barplot(table(heart_failure$sex))
hist(heart_failure$age)

# Split into training and testing -----------------------------------------

set.seed(20231018)
hf_split <- initial_split(heart_failure)
hf_train <- training(hf_split)
hf_test <- testing(hf_split)

# choose a different split proportion?
set.seed(20231018)
hf_split <- initial_split(heart_failure, prop = 0.8)
hf_train <- training(hf_split)
hf_test <- testing(hf_split)

# Create cross validation folds
hf_folds <- vfold_cv(hf_train, v = 10)


# Build a recipe ----------------------------------------------------------

hf_recipe <- recipe(death ~ ., data = hf_train) |>
  step_dummy(all_nominal_predictors()) |>  # Convert all categorical variables to dummies
  step_normalize(age, serum_creatinine, creatinine_phosphokinase, 
                 platelets, ejection_fraction, time) 

wf <- workflow() |> 
  add_recipe(hf_recipe)

# Specify the model -------------------------------------------------------

tune_spec_lasso <- logistic_reg(penalty = tune(), mixture = 1) |>
  set_engine("glmnet")

# Tune the model ----------------------------------------------------------

# Fit lots of values
lasso_grid <- tune_grid(
  add_model(wf, tune_spec_lasso),
  resamples = hf_folds,
  grid = grid_regular(penalty(), levels = 50)
)

# Choose the best value
highest_roc_auc_lasso <- lasso_grid |>
  select_best("roc_auc")


# Fit the final model -----------------------------------------------------

final_lasso <- finalize_workflow(
  add_model(wf, tune_spec_lasso),
  highest_roc_auc_lasso
)