#### AGDS II: Digital Soil Mapping Exercise 5.1
### Load Data
df_full <- readRDS(here::here("data/df_full.rds"))

head(df_full) |>
  knitr::kable()

## Specify target: Waterlog.100
target <- "waterlog.100"

## Specify predictors_all: Remove soil sampling and observational data
predictors_all <- names(df_full)[14:ncol(df_full)]

cat("The target is:", target,
    "\nThe predictors_all are:", paste0(predictors_all[1:8], sep = ", "), "...")

# Split dataset into training and testing sets
df_train <- df_full |> dplyr::filter(dataset == "calibration")
df_test  <- df_full |> dplyr::filter(dataset == "validation")

# Filter out any NA to avoid error when running a Random Forest
df_train <- df_train |> tidyr::drop_na()
df_test <- df_test   |> tidyr::drop_na()

# A little bit of verbose output:
n_tot <- nrow(df_train) + nrow(df_test)

perc_cal <- (nrow(df_train) / n_tot) |> round(2) * 100
perc_val <- (nrow(df_test)  / n_tot) |> round(2) * 100

cat("For model training, we have a calibration / validation split of: ",
    perc_cal, "/", perc_val, "%")

### Train Model
rf_basic <- ranger::ranger(
  probability = FALSE,
  classification = TRUE,
  y = df_train[, target],     # target variable
  x = df_train[, predictors_all], # Predictor variables
  seed = 42,                    # Specify the seed for randomization to reproduce the same model again
  num.threads = parallel::detectCores() - 1) # Use all but one CPU core for quick model training

# Print a summary of fitted model
print(rf_basic)

#Save Data
# Save relevant data for model testing in the next chapter.
saveRDS(rf_basic,
        here::here("data/rf_basic_for_waterlog100.rds"))

saveRDS(df_train[, c(target, predictors_all)],
        here::here("data/cal_basic_for_waterlog100.rds"))

saveRDS(df_test[, c(target, predictors_all)],
        here::here("data/val_basic_for_waterlog100.rds"))

#### Model Analysis
# Load random forest model
rf_basic   <- readRDS(here::here("data/rf_basic_for_waterlog100.rds"))
df_train <- readRDS(here::here("data/cal_basic_for_waterlog100.rds"))
df_test  <- readRDS(here::here("data/val_basic_for_waterlog100.rds"))

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

# Filter for variables used in RF
preds_all <- names((df_train[, predictors_all]))
files_selected <- files_covariates[apply(sapply(X = preds_all,
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
  rf_basic,           # RF model
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
conf_matrix_waterlog_rfbasic <- caret::confusionMatrix(data=X, reference=Y, positive="1")
conf_matrix_waterlog_rfbasic

mosaicplot(conf_matrix_waterlog_rfbasic$table,
           main = "Confusion matrix")

## Accuracy of 0.75

### Create Prediction Maps
prediction <- predict(
  rf_basic,              # RF model
  data = df_predict,
  num.threads = parallel::detectCores() - 1)

# Attach predictions to dataframe and round them
df_predict$prediction <- prediction$predictions
