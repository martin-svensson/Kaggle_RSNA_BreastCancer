# ====================================================================================================== #
# Description
#
#   Develop code chunks for the model notebook using a small subset of patients (two without cancer, one with)
#
# Change log:
#   Ver   Date        Comment
#   1.0   09/12/22    Initial version
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
library(ptw) # padding pixel matrices
library(imager)
library(keras)
library(tensorflow)
library(reticulate)
# c("pydicom", "python-gdcm", "pylibjpeg") %>% 
#   py_install(pip = TRUE)

pydicom <- import("pydicom") # there are no R packages which work with JPEG2000
pylibjpeg <- import("pylibjpeg")
gdcm <- import("gdcm")

pyraug <- import("pyraug") # for synthetic oversampling
torch <- import("torch")

# ------------------------------------------------------------------------------------------------------ #
# IMPORT AND SOURCES
# ------------------------------------------------------------------------------------------------------ #

test_patients <- 
  list.dirs(
    "./Data/train_images/", 
    full.names = FALSE, 
    recursive = FALSE
  ) %>% as.integer()

df_data <- fread("./Data/train.csv")

# ------------------------------------------------------------------------------------------------------ #
# PROGRAM
# ------------------------------------------------------------------------------------------------------ #

# ---- Load images -------------------------------------------------------------

# -- Function to rescale images such that they can be stored as sparse matrices

fun_rescale <- function(dcm) {
  
  img <- dcm$pixel_array
  
  if (dcm$PhotometricInterpretation == "MONOCHROME1") {
    # MONOCROME1 is bright to dark. We want dark to bright (like MONOCRHOME2), so we can store it as a sparse matrix
    
    img <-  max(img) - img 
    
  }
  
  slope <- as.double(as.character(dcm$RescaleSlope))
  intercept <- as.double(as.character(dcm$RescaleIntercept))
  
  result <- img * slope + intercept
  
  result <- result - min(result)
  
  result <- result / max(result) # normalize 
  
  return(result)
  
}

# -- Subset df_data to keep test_patients

df_data %<>% 
  filter(patient_id %in% test_patients)

# -- add image path

df_data %<>% 
  mutate(
    dcm_path = paste0("./Data/train_images/", patient_id, "/", image_id, ".dcm")
  )

# -- rescaled pixel data 

dcm_image <- 
  map(
    df_data$dcm_path,
    ~ pydicom$dcmread(.x) %>% fun_rescale()
  ) 

# pydicom is probably a bit too slow when we run this on all images, so we might have to 
# use the byte data directly as in https://www.kaggle.com/code/tivfrvqhs5/decode-jpeg2000-dicom-with-dali?scriptVersionId=113466193
# use test_dcm[[10]]$file_meta$TransferSyntaxUID to see the file type
# 
# test <- espadon::dicom.raw.data.loader(df_data$dcm_path[15])
# test2 <- test[test != "00"] %>% rawToChar()
# 
# test2 %>% str_locate("\\x0C")
# 
# test2 %>% str_sub(1, 1000)


# -- test how images look

if (FALSE) {
# this is to avoid running the following chunk when running script from the top
  
  img <- dcm_image[[15]]
  jpeg(filename = "./Output/test.jpg")
  image(
    t(img[nrow(img):1, ]), 
    col = hcl.colors(12, palette = "Grays", rev = FALSE), 
    useRaster = TRUE
  )
  dev.off()

}



# ---- Resampling --------------------------------------------------------------

data_split <- 
  initial_split(
    df_train, 
    prop = 0.8,
    strata = cancer
  )

df_train <- training(data_split)
df_test <- testing(data_split)

set.seed(48653)
df_train_folds <- 
  vfold_cv(
    df_train,
    v = 5,
    strata = cancer
  )


# ---- Prepare images ----------------------------------------------------------

# -- Crop images 

fun_crop_image <- function(img, threshold) {
    
  zero_rows <- 
    img %>% 
    rowSums() %>%  
    {. > threshold}
  
  zero_cols <- 
    img %>% 
    colSums() %>%  
    {. > threshold}
  
  return(img[zero_rows, zero_cols])
  
}

threshold_try <- 60 # 100 seems to crop the images well, however, we might expore other values (perhaps even tuning it). 

dcm_image %<>% 
  map(
    ~ fun_crop_image(.x, threshold_try)
  )

# -- Reduce resolution

fun_resize <- function(img, target_size, interpolation = 6) {
  
  placeholder <- array(dim = c(nrow(img), ncol(img), 1, 1)) # the last two dim are for color channels and video 
  
  placeholder[ , , 1, 1] <- img
  
  scale <- target_size / nrow(img)
  
  result <-
    placeholder %>% 
    imresize(
      scale,
      interpolation
    )
  
  return(as.matrix(result))
  
}

target_size_try <- 560

dcm_image %<>% 
  map(
    ~ fun_resize(.x, target_size = target_size_try)
  )

# -- Pad imges to same size
# we pad the images to maintain aspect ratio (the alternative is to resize all of them to the same size)

# max_pixel_x <- 
#   map_int(
#     dcm_image,
#     ~ .x %>% nrow
#   ) %>% max()
#  WE USE TARGET_SIZE INSTEAD

max_pixel_y <- 
  map_int(
    dcm_image,
    ~ .x %>% ncol
  ) %>% max()

max_pixel_y <- 
  max_pixel_y + 28 - (max_pixel_y %% 28) # the autoencoder needs to be able to recreate the correct number of pixels

fun_pad_image <- function(img) {
  
  # pad left
  img %<>% {
    padzeros(
      data = .,
      nzeros = max_pixel_y - ncol(.),
      side = "left"
    )
  }
  
  # pad top
  img %<>% {
    padzeros(
      data = t(.),
      nzeros = target_size_try - nrow(.),
      side = "left"
    )
  } %>% t()
  
  return(img)
  
}

dcm_image %<>% 
  map(
    ~ fun_pad_image(.x)
  )

# keras accepts 3d arrays, not lists
dcm_image %<>% 
  simplify2array() %>% 
  aperm(c(3, 1, 2))


# ---- Class imbalance and augmentation ----------------------------------------


# -- Downsampling ------------------------------------------------------------

# Downsample from images which are clearly non-cancerous. By "cearly" we
# might fx mean cancer = 0 with BIRADS %in% 1 or 2 and difficult_negative_case = FALSE

# We might want to do this as a very first step to reduce the amount of data from the get go


# -- Upsampling and augmentation ---------------------------------------------


# VAE ---------------------------------------------------------------------
# https://github.com/clementchadebec/pyraug/blob/main/examples/getting_started.ipynb

# -- Fit the VAE

TrainingConfig <- py_run_string("from pyraug.trainers.training_config import TrainingConfig")
TrainingPipeline <- py_run_string("from pyraug.pipelines.training import TrainingPipeline")

config <- 
  TrainingConfig$TrainingConfig(
    output_dir="./Output/pyraug",
    # train_early_stopping=50,
    # learning_rate=1e-3,
    # batch_size=50, 
    max_epochs=150
  )

pipeline <- 
  TrainingPipeline$TrainingPipeline(
    training_config=config
  )

pipeline(
  train_data = dcm_image,
  log_output_dir = "./Output/pyraug/"
)

# -- Generation pipeline

GenerationPipeline <- py_run_string("from pyraug.pipelines.generation import GenerationPipeline")
RHVAE <- py_run_string("from pyraug.models import RHVAE")

model_path <- 
  list.dirs("./Output/pyraug") %>% 
  grep("final_model", x = ., value = TRUE) %>% 
  {.[length(.)]}

model <- RHVAE$RHVAE$load_from_folder(model_path)

generation_pipeline <- 
  GenerationPipeline$GenerationPipeline(model = model)

# -- generate images

generation_pipeline(100)

gen_path <- 
  list.dirs(
    "./dummy_output_dir", 
    recursive = FALSE
  ) %>% {.[length(.)]}

generated_data <- torch$load(paste0(gen_path, "/generated_data_100_0.pt"))

test <- generated_data[0]$cpu()$reshape(as.integer(target_size_try), as.integer(max_pixel_y))$numpy()
test2 <- test %>% as.matrix()


# My autoencoder ----------------------------------------------------------
# based on: https://keras.io/examples/vision/autoencoder/
# The issue is that regular autoencoders are not great at generating augmented images (they are better at denoising etc)

if (FALSE) { # to avoid accidentally running it
  
  autoenc <- keras_model_sequential()
  
  autoenc %>% 
    layer_conv_2d(
      filters = 8,
      kernel_size = c(3, 3),
      padding = "same",
      activation = "relu",
      input_shape = c(target_size_try, max_pixel_y, 1)
    ) %>% 
    layer_max_pooling_2d() %>%   
    layer_conv_2d(
      filters = 8,
      kernel_size = c(3, 3),
      padding = "same",
      activation = "relu"
    ) %>% 
    layer_max_pooling_2d() %>%   
    layer_conv_2d_transpose(
      filters = 8,
      kernel_size = c(3, 3),
      strides = 2, 
      activation = "relu",
      padding = "same"
    ) %>%
    layer_conv_2d_transpose(
      filters = 8,
      kernel_size = c(3, 3),
      strides = 2, 
      activation = "relu",
      padding = "same"
    ) %>% 
    layer_conv_2d(
      filters = 1,
      kernel_size = c(3, 3),
      padding = "same",
      activation = "sigmoid"
    )
    
  autoenc %>% 
    compile(
      optimizer = "adam",
      loss = "binary_crossentropy"
    )
  
  # -- Train autoencoder
  # use a small subset of data for training
  
  n_train <- 10
  
  autoenc %>% 
    fit(
      x = dcm_image,
      y = dcm_image,
      epochs = 80,
      verbose = 2
    )
  
  # -- Validation
  
  test <- 
    autoenc %>% 
    predict(dcm_image)

}


# ---- Model -------------------------------------------------------------------

model <- keras_model_sequential()

model %>% 
  layer_conv_2d(
    filters = 16,
    kernel_size = c(3, 3),
    padding = "valid",
    activation = "relu",
    input_shape = c(max_pixel_x, max_pixel_y, 1)
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
  layer_flatten(
    input_shape = c(max_pixel_x, max_pixel_y)
  ) %>% 
  layer_dense(
    units = 10,
    activation = "relu"
  ) %>% 
  layer_dense(
    units = 1, 
    activation = "sigmoid",
    name = "Output"
  )

model %>% 
  compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )

# -- Train model

n_train <- 12

model %>% 
  fit(
    dcm_image[1:n_train, , ],
    df_train$cancer[1:n_train],
    epochs = 8,
    verbose = 2
  )

# -- Validation

model %>% 
  predict(dcm_image[(n_train + 1):15, , ])







# ==== EXPORT ------------------------------------------------------------------------------------------ 