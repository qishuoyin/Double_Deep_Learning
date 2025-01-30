# Load necessary libraries
library(httr)       
library(tidyverse)  
library(haven)      
library(janitor)    
library(fastDummies)
install.packages("SASxport")
library(SASxport)

# ✅ Step 1: Verify NHANES Dataset URLs (Manually Check!)
nhanes_urls <- list(
  "DEMO" = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO.XPT",
  "BMX_J" = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT",
  "BPX_J" = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT",
  "DIQ_J" = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DIQ_J.XPT",
  "HDL_J" = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/HDL_J.XPT",
  "TRIGLY_J" = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/TRIGLY_J.XPT"
  # 🔹 Add more datasets manually after verifying URLs
)

# ✅ Step 2: Set Up Download Directory
data_dir <- "nhanes_data"
if (!dir.exists(data_dir)) dir.create(data_dir)

# ✅ Step 3: Delete & Re-Download Corrupt `.XPT` Files
download_nhanes_data <- function(dataset_name, url) {
  file_path <- file.path(data_dir, paste0(dataset_name, ".XPT"))
  
  # Delete if it's an error page
  if (file.exists(file_path)) {
    file_head <- readLines(file_path, n = 10, warn = FALSE)
    if (any(grepl("404|Error|Not Found|Access Denied|<!DOCTYPE html>", file_head))) {
      message(paste("🚨 Corrupt file detected:", dataset_name, "- Deleting and re-downloading..."))
      file.remove(file_path)
    }
  }
  
  # Download if file does not exist or was deleted
  if (!file.exists(file_path)) {
    tryCatch({
      message("🔽 Downloading:", dataset_name)
      download.file(url, file_path, mode = "wb")
      message("✅ Successfully downloaded:", dataset_name)
    }, error = function(e) {
      message("❌ Failed to download:", dataset_name, "- Check URL manually.")
    })
  } else {
    message(paste("⏩ Already exists:", dataset_name))
  }
}

# Run the re-download process
invisible(mapply(download_nhanes_data, names(nhanes_urls), nhanes_urls))

# ✅ Step 4: Use `SASxport` to Read `.XPT` Files
xpt_files <- list.files(data_dir, pattern = "\\.XPT$", full.names = TRUE)
if (length(xpt_files) == 0) stop("❌ No .XPT files found. Please check the download process.")

read_nhanes_data <- function(file) {
  tryCatch({
    read.xport(file) %>%
      clean_names() %>%
      mutate(dataset_name = tools::file_path_sans_ext(basename(file)))
  }, error = function(e) {
    message("❌ Failed to read:", file)
    return(NULL)
  })
}

# Read all datasets
nhanes_list <- lapply(xpt_files, read_nhanes_data)
nhanes_list <- nhanes_list[!sapply(nhanes_list, is.null)]
if (length(nhanes_list) == 0) stop("❌ No valid datasets were read.")

# ✅ Step 5: Merge All NHANES Datasets
if (length(nhanes_list) == 1) {
  merged_data <- nhanes_list[[1]]
} else {
  merged_data <- reduce(nhanes_list, full_join, by = "seqn")
}

message("✅ Successfully merged NHANES datasets!")

# ✅ Step 6: Data Cleaning
merged_data <- merged_data %>%
  select(-contains(".x"), -contains(".y")) %>%
  distinct()

# Remove rows where more than 50% of values are missing
na_threshold <- 0.5 * ncol(merged_data)
cleaned_data <- merged_data[rowSums(is.na(merged_data)) <= na_threshold, ]

# Replace remaining NA values with "Missing"
cleaned_data <- cleaned_data %>%
  mutate_all(~ ifelse(is.na(.), "Missing", .))

# ✅ Step 7: Convert Categorical Variables into Dummy Variables
categorical_vars <- cleaned_data %>%
  select_if(is.character) %>%
  colnames()

cleaned_data <- dummy_cols(cleaned_data, select_columns = categorical_vars, 
                           remove_first_dummy = TRUE, remove_selected_columns = TRUE)

message("✅ Categorical variables converted to dummy variables!")

# ✅ Step 8: Save the Final Cleaned Dataset
write.csv(cleaned_data, "nhanes_data/NHANES_2017_2018_Final_Cleaned.csv", row.names = FALSE)
message("🎉 Data processing complete! Cleaned dataset saved in: nhanes_data/NHANES_2017_2018_Final_Cleaned.csv")

