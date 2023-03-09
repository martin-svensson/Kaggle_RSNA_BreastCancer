# ====================================================================================================== #
# Description
#
#   Save images as png
#
# Change log:
#   Ver   Date        Comment
#   1.0   23/12/22    Initial version
#
# ====================================================================================================== #
# ------------------------------------------------------------------------------------------------------ #
# LIBRARIES
# ------------------------------------------------------------------------------------------------------ #

library(data.table)
library(tidyverse)
library(magrittr)

library(imager)

library(reticulate)
pydicom <- import("pydicom") # there are no R packages which work with JPEG2000
pylibjpeg <- import("pylibjpeg")
gdcm <- import("gdcm")

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

# ---- functions for processing ---------------------------------------------------------------

fun_rescale <- function(dcm) {
  
  img <- dcm$pixel_array
  
  if (dcm$PhotometricInterpretation == "MONOCHROME1") {
    # MONOCROME1 is bright to dark. We want dark to bright (like MONOCRHOME2), so we can store it as a sparse matrix
    
    img <-  max(img) - img 
    
  }
  
  img <- (img - min(img)) / (max(img) - min(img))
  
  return(img)
  
}


fun_dark_pixels <- function(img, threshold = 0.5) {
  
  img[which(img < threshold)] <- 0
  
  return(img)
  
}
  

fun_resize <- function(img, target_size, interpolation = 6) {
  # input: pixel matrix
  # output: imager::cimg
  
  tmp <- img %>% as.cimg()
  
  result <-
    tmp %>% 
    resize(
      size_x = target_size,
      size_y = target_size,
      interpolation_type = interpolation
    )

  return(result)
  
}


# ---- Load an process images --------------------------------------------------

df_data %<>%
  filter(
    patient_id %in% patients
  ) %>% 
  mutate(
    dcm_path = paste0("./Data/train_images/", patient_id, "/", image_id, ".dcm")
  )

dcm_image <- 
  map(
    df_data$dcm_path,
    ~ pydicom$dcmread(.x) %>% 
      fun_rescale() %>% 
      fun_dark_pixels() %>% 
      fun_resize(target_size = 512)
  ) 

names(dcm_image) <- 
  paste0(df_data$patient_id, "_", df_data$image_id)


# ==== EXPORT ------------------------------------------------------------------------------------------ 

walk2(
  .x = dcm_image,
  .y = names(dcm_image),
  ~ save.image(.x, file = paste0("./Output/images/", .y, ".png"))
)
  
