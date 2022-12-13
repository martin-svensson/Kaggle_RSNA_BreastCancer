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
library(ptw) # padding pixel matrices
library(keras)
library(tensorflow)
library(reticulate)
# c("pydicom", "python-gdcm", "pylibjpeg") %>% 
#   py_install(pip = TRUE)

pydicom <- import("pydicom") # there are no R packages which work with JPEG2000
pylibjpeg <- import("pylibjpeg")
gdcm <- import("gdcm")

# ------------------------------------------------------------------------------------------------------ #
# IMPORT AND SOURCES
# ------------------------------------------------------------------------------------------------------ #

test_patients <- 
  list.dirs(
    "./Data/train_images/", 
    full.names = FALSE, 
    recursive = FALSE
  ) %>% as.integer()

df_train <- fread("./Data/train.csv")
df_test <- fread("./Data/test.csv")

# ------------------------------------------------------------------------------------------------------ #
# PROGRAM
# ------------------------------------------------------------------------------------------------------ #

df_train %<>% 
  as_tibble()

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
  
  result <- result / max(result) # normalize 
  
  return(result)
  
}

# -- Subset df_train to keep test_patients

df_train %<>% 
  filter(patient_id %in% test_patients)

# -- add image path

df_train %<>% 
  mutate(
    dcm_path = paste0("./Data/train_images/", patient_id, "/", image_id, ".dcm")
  )

# -- rescaled pixel data as 3d array (for keras)

dcm_image <- 
  map(
    df_train$dcm_path,
    ~ pydicom$dcmread(.x) %>% fun_rescale()
  ) 

# pydicom is probably a bit too slow when we run this on all images, so we might have to 
# use the byte data directly as in https://www.kaggle.com/code/tivfrvqhs5/decode-jpeg2000-dicom-with-dali?scriptVersionId=113466193
# use test_dcm[[10]]$file_meta$TransferSyntaxUID to see the file type
# 
# test <- espadon::dicom.raw.data.loader(df_train$dcm_path[15])
# test2 <- test[test != "00"] %>% rawToChar()
# 
# test2 %>% str_locate("\\x0C")
# 
# test2 %>% str_sub(1, 1000)


# -- test how images look

img <- dcm_image_cropped[[3]]
jpeg(filename = "./Output/test.jpg")
image(
  t(img[nrow(img):1, ]), 
  col = hcl.colors(12, palette = "Grays", rev = FALSE), 
  useRaster = TRUE
)
dev.off()


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

threshold_try <- 100 # 100 seems to crop the images well, however, we might expore other values (perhaps even tuning it). 

dcm_image %<>% 
  map(
    ~ fun_crop_image(.x, 100)
  )


# -- Pad imges to same size

max_pixel_x <- 
  map_int(
    dcm_image,
    ~ .x %>% nrow
  ) %>% max()

max_pixel_y <- 
  map_int(
    dcm_image,
    ~ .x %>% ncol
  ) %>% max()

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
      nzeros = max_pixel_x - nrow(.),
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




# ---- Model -------------------------------------------------------------------

model <- keras_model_sequential()

model %>% 
  layer_flatten(input_shape = c(max_pixel_x, max_pixel_y)) %>% 
  layer_dense(
    units = 10,
    activation = "relu"
  ) %>% 
  layer_dense(
    units = 1, 
    activation = "sigmoid"
  )

model %>% 
  compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )

n_train <- 10

model %>% 
  fit(
    dcm_image[1:n_train, , ],
    df_train$cancer[1:n_train],
    epochs = 5,
    verbose = 2
  )


# -- Predictions

model %>% 
  predict(dcm_image[(n_train + 1):15, , ])







# ==== EXPORT ------------------------------------------------------------------------------------------ 