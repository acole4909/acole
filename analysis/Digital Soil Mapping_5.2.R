##### Exercise 5: 5.2
#### Data Loading and Preparation
### Load Data
df_full <- readRDS(here::here("data/df_full.rds"))

head(df_full) |>
  knitr::kable()

### Variable Importance
rf_varimport <- ranger::ranger(
  probability = FALSE,
  y = df_train[, target],     # target variable
  x = df_train[, predictors_all],   # Predictor variables
  importance   = "impurity", # Pick permutation to calculate variable importance
  classification = TRUE,
  seed = 42,                    # Specify seed for randomization to reproduce the same model again
  num.threads = parallel::detectCores() - 1) # Use all but one CPU core for quick model training

print(rf_varimport)

# Extract the variable importance and create a long tibble
vi_rf_varimport <- rf_varimport$variable.importance |>
  dplyr::bind_rows() |>
  tidyr::pivot_longer(cols = dplyr::everything(), names_to = "variable")

# Plot variable importance, ordered by decreasing value
gg <- vi_rf_varimport |>
  ggplot2::ggplot(ggplot2::aes(x = reorder(variable, value), y = value)) +
  ggplot2::geom_bar(stat = "identity", fill = "grey50", width = 0.75) +
  ggplot2::labs(
    y = "Change in OOB MSE after permutation",
    x = "",
    title = "Variable importance based on OOB") +
  ggplot2::theme_classic() +
  ggplot2::coord_flip()

# Display plot
gg

### 3.5 Variable Selection
set.seed(42)

# run the algorithm
bor <- Boruta::Boruta(
  y = df_train[, target],
  x = df_train[, predictors_all],
  maxRuns = 50, # Number of iterations. Set to 30 or lower if it takes too long
  num.threads = parallel::detectCores()-1)

# obtain results: a data frame with all variables, ordered by their importance
df_bor <- Boruta::attStats(bor) |>
  tibble::rownames_to_column() |>
  dplyr::arrange(dplyr::desc(meanImp))

# plot the importance result
ggplot2::ggplot(ggplot2::aes(x = reorder(rowname, meanImp),
                             y = meanImp,
                             fill = decision),
                data = df_bor) +
  ggplot2::geom_bar(stat = "identity", width = 0.75) +
  ggplot2::scale_fill_manual(values = c("grey30", "tomato", "grey70")) +
  ggplot2::labs(
    y = "Variable importance",
    x = "",
    title = "Variable importance based on Boruta") +
  ggplot2::theme_classic() +
  ggplot2::coord_flip()

# get retained important variables
predictors_selected <- df_bor |>
  dplyr::filter(decision == "Confirmed") |>
  dplyr::pull(rowname)

length(predictors_selected)

# re-train Random Forest model
rf_bor <- ranger::ranger(
  probability = FALSE,
  y = df_train[, target],              # target variable
  x = df_train[, predictors_selected], # Predictor variables
  classification = TRUE,
  seed = 42,                           # Specify the seed for randomization to reproduce the same model again
  num.threads = parallel::detectCores() - 1) # Use all but one CPU core for quick model training

# quick report and performance of trained model object
rf_bor

# Save relevant data for model testing in the next chapter.
saveRDS(rf_bor,
        here::here("data/rf_bor_for_waterlog.100.rds"))

saveRDS(df_train[, c(target, predictors_selected)],
        here::here("data/cal_bor_for_waterlog.100.rds"))

saveRDS(df_test[, c(target, predictors_selected)],
        here::here("data/val_bor_for_waterlog.100.rds"))

# Load random forest model
rf_bor   <- readRDS(here::here("data/rf_bor_for_waterlog.100.rds"))
df_train <- readRDS(here::here("data/cal_bor_for_waterlog.100.rds"))
df_test  <- readRDS(here::here("data/val_bor_for_waterlog.100.rds"))

# Load area to be predicted
raster_mask <- terra::rast(here::here("data-raw/geodata/study_area/area_to_be_mapped.tif"))
# Turn target raster into a dataframe, 1 px = 1 cell
df_mask <- as.data.frame(raster_mask, xy = TRUE)

# Filter only for area of interest
df_mask <- df_mask |>
  dplyr::filter(area_to_be_mapped == 1)

# Display df
head(df_mask) |>
  knitr::kable()

files_covariates <- list.files(
  path = here::here("data-raw/geodata/covariates/"),
  pattern = ".tif$",
  recursive = TRUE,
  full.names = TRUE
)

random_files <- sample(files_covariates, 2)
terra::rast(random_files[1])
terra::rast(random_files[2])

# Filter that list only for the variables used in the RF
preds_selected <- names(df_train[, predictors_selected])
files_selected <- files_covariates[apply(sapply(X = preds_selected,
                                                FUN = grepl,
                                                files_covariates),
                                         MARGIN =  1,
                                         FUN = any)]

# Load all rasters as a stack
raster_covariates <- terra::rast(files_selected)

# Get coordinates for which we want data
df_locations <- df_mask |>
  dplyr::select(x, y)

# Extract data from covariate raster stack for all gridcells in the raster
df_predict <- terra::extract(
  raster_covariates,   # The raster we want to extract from
  df_locations,        # A matrix of x and y values to extract for
  ID = FALSE           # To not add a default ID column to the output
)

df_predict <- cbind(df_locations, df_predict) |>
  tidyr::drop_na()  # Se_TWI2m has a small number of missing data

# Make predictions for validation sites
prediction <- predict(
  rf_bor,           # RF model
  data = df_test,   # Predictor data
  num.threads = parallel::detectCores() - 1
)

# Save predictions to validation df
df_test$pred <- prediction$predictions

# Classification Metrics
Y <- df_test$waterlog.100
Y <- as.factor(Y)
X <- df_test$pred
X <- as.factor(X)
library(caret)
#Confusion Matrix
conf_matrix_waterlog_bor <- caret::confusionMatrix(data=X, reference=Y)
conf_matrix_waterlog_bor

## Accuracy increases to 0.79 as compared to accuracy of 0.75 for rf_basic. Based on this metric, the rf_bor performs better.
## The OOB prediction error reported by rf_basic was 21.32% while the OOB prediction error for rf_bor was 20.99%.

## Create ROC curve with probabilities for target
