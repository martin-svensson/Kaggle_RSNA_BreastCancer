# ====================================================================================================== #
# Description
#
#   Create image model for sample images (no down- and upsampling is carried out)
#
# Change log:
#   Ver   Date        Comment
#   1.0   25/12/22    Initial version
#
# ====================================================================================================== #
# ------------------------------------------------------------------------------------------------------ #
# LIBRARIES
# ------------------------------------------------------------------------------------------------------ #

library(data.table)
library(tidyverse)
library(magrittr)

# -- Specific libs
library(tidymodels)
library(imager)
library(keras)
library(tensorflow)
library(reticulate)

pyraug <- import("pyraug") # for synthetic oversampling
torch <- import("torch")

# ------------------------------------------------------------------------------------------------------ #
# IMPORT AND SOURCES
# ------------------------------------------------------------------------------------------------------ #

patients <- 
  list.dirs(
    "./Data/train_images/", 
    full.names = FALSE, 
    recursive = FALSE
  ) %>% as.integer()

df_data <- fread("./Data/train.csv")

# ------------------------------------------------------------------------------------------------------ #
# PROGRAM
# ------------------------------------------------------------------------------------------------------ #

# -- Create dirs for positive and negative cases (for flow_images_from_directory())

dir_positive_cases <- "./Output/positive_cases"
dir_negative_cases <- "./Output/negative_cases"

if (!dir.exists(dir_positive_cases)) {
  dir.create(dir_positive_cases)
}

if (!dir.exists(dir_negative_cases)) {
  dir.create(dir_negative_cases)
}

# -- Add image path

df_data %<>%
  filter(
    patient_id %in% patients
  ) %>% 
  mutate(
    from_path = paste0("./Output/images/", patient_id, "_", image_id, ".png"),
    to_path = 
      if_else(
        cancer == 1, 
        paste0(dir_positive_cases, "/", patient_id, "_", image_id, ".png"), 
        paste0(dir_negative_cases, "/", patient_id, "_", image_id, ".png")
      )
  )

# -- copy images to class subdir

file.copy(
  from = df_data$from_path,
  to = df_data$to_path
) %>% invisible()


# ---- Model -------------------------------------------------------------------

target_size <- c(512, 512)
batch_size <- 32

ker_image_from_dir <- 
  flow_images_from_directory(
    directory = "./Output/",
    classes = c("positive_cases", "negative_cases"),
    class_mode = "binary",
    target_size = target_size,
    color_mode = "grayscale",
    batch_size = batch_size,
    seed = 38542
  )
  
tf$random$set_seed(681) # initial parameters

model <- keras_model_sequential()

model %>% 
  layer_conv_2d(
    filters = 16,
    kernel_size = c(3, 3),
    padding = "valid",
    activation = "relu",
    input_shape = c(target_size, 1)
  ) %>% 
  layer_max_pooling_2d() %>% 
  layer_conv_2d(
    filters = 20,
    kernel_size = c(3, 3),
    padding = "valid",
    activation = "relu"
  ) %>% 
  layer_max_pooling_2d() %>% 
  layer_conv_2d(
    filters = 24,
    kernel_size = c(3, 3),
    padding = "valid",
    activation = "relu"
  ) %>%
  layer_max_pooling_2d() %>%
  layer_conv_2d(
    filters = 28,
    kernel_size = c(3, 3),
    padding = "valid",
    activation = "relu"
  ) %>%
  layer_max_pooling_2d() %>%
  layer_conv_2d(
    filters = 32,
    kernel_size = c(3, 3),
    padding = "valid",
    activation = "relu"
  ) %>%
  layer_max_pooling_2d() %>%
  layer_flatten() %>% 
  layer_dense(
    units = 10,
    activation = "relu"
  ) %>% 
  layer_dense(
    units = 1, 
    activation = "sigmoid",
    name = "Output"
  )

# -- Compile

model %>% 
  compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )

# -- Fit

model_fit <-
  model %>% 
  fit(
    ker_image_from_dir,
    epochs = 5,
    verbose = 2
  )

# -- Predict

model %>% 
  predict(ker_image_from_dir)

ker_image_from_dir$classes

# ==== EXPORT ------------------------------------------------------------------------------------------ 