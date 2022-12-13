# ====================================================================================================== #
# Description
#
#   Code chunks for EDA
#
# Change log:
#   Ver   Date        Comment
#   1.0   07/12/22    Initial version
#
# ====================================================================================================== #
# ------------------------------------------------------------------------------------------------------ #
# LIBRARIES
# ------------------------------------------------------------------------------------------------------ #

# -- Standard libs
library(data.table)
library(tidyverse)
library(magrittr)
library(DataExplorer)

# -- Specific libs
library(reticulate)
pydicom <- import("pydicom") # there are no R packages which work with JPEG2000
library(SparseM)
library(grDevices)

# ------------------------------------------------------------------------------------------------------ #
# IMPORT AND SOURCES
# ------------------------------------------------------------------------------------------------------ #

patient_id <- "10006"

patient_dcm <- 
  list.files(
    paste0("./Data/train_images/", patient_id), 
    full.names = TRUE
  ) %>% 
  purrr::set_names(
    nm = str_extract(string = ., "\\d{1,}(?=\\.dcm)")
  ) %>% 
  map(
    ~ pydicom$dcmread(.x)
  )

df_train <- fread("./Data/train.csv")
df_test <- fread("./Data/test.csv")

# ------------------------------------------------------------------------------------------------------ #
# PROGRAM
# ------------------------------------------------------------------------------------------------------ #

# -- Metdata 

patient_dcm[[1]]$elements

# -- Image

fun_rescale <- function(dcm) {
  
  img <- dcm$pixel_array
  
  if (dcm$PhotometricInterpretation == "MONOCHROME1") {
    
    img <-  max(img) - img 
    
  } else {
    
    stop("Img is not MONOCHROME1")
    
  }
  
  slope <- as.double(as.character(dcm$RescaleSlope))
  intercept <- as.double(as.character(dcm$RescaleIntercept))
  
  result <- img * slope + intercept
  
  return(result)
  
}

img <- 
  patient_dcm[[2]] %>% 
  fun_rescale() # super large image files

img_sparse <- 
  as.matrix.csr(img) # doesn't plot well for some reason


jpeg(filename = "./Output/test.jpg")
image(
  t(img[nrow(img):1, ]), 
  col = hcl.colors(12, palette = "Grays", rev = FALSE), 
  useRaster = TRUE
)
dev.off()

# -- patients and images

df_images <- 
  tibble(
    patient_id = list.files("./Data/train_images"),
    n_images = 
      map_int(
        patient_id,
        ~ list.files(paste0("./Data/train_images/", .x)) %>% length
      )
  )

df_images %>% 
  count(n_images) %>% 
  ggplot(aes(x = n_images)) +
  geom_bar() +
  scale_x_discrete()



# ---- Metadata ----------------------------------------------------------------

label <- "cancer"

var_id <- c("patient_id", "image_id")

var_num <- "age"

var_factor <- 
  setdiff(names(df_train), c(var_id, var_num))

var_only_train <- 
  c("density", "biopsy", "invasive", "BIRADS", "difficult_negative_case")

df_train[
  ,
  (var_factor) := map(.SD, as.factor), 
  .SDcols = var_factor
]

# -- missing values

df_train %>%
  plot_missing()

# -- categorical features and response

df_train %>% 
  select(-label) %>% 
  plot_bar

df_train %>% 
  plot_bar(by = "cancer")

# -- numerical features

df_train %>% 
  ggplot(aes(x = age, fill = cancer)) +
  geom_density(alpha = 0.4)

# -- pairwise correlations

df_train %>% 
  select(!c(var_id, "machine_id", var_only_train)) %>% 
  plot_correlation(
    type = "discrete", 
    cor_args = list("method" = "spearman")
  )




# ==== EXPORT ------------------------------------------------------------------------------------------ 