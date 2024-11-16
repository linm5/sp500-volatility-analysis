library(readxl)
input_file <- "/Users/meishanlin/BDA_project/BDA_project/Price History_20241108_1323.xlsx" 
output_file <- "/Users/meishanlin/BDA_project/BDA_project/SP500_History.csv.gz"
data <- read_excel(input_file, skip = 17)  
write.csv(data, gzfile(output_file), row.names = FALSE)
cat("Filtered data saved to:", output_file, "\n")
