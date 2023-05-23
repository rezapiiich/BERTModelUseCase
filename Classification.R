df <- read.csv("joined_clean.csv")

# Get the unique values of issue_map
unique_issue_vals <- unique(df$issue_map)

english_rows <- df[grepl("a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z", df$Comment, ignore.case = TRUE), ]

# Loop over each unique value and replace it in the Comment column
for (val in unique_issue_vals) {
  df$Comment <- gsub(val, "", df$Comment, ignore.case = TRUE)
}
rm(val)

df$Comment <- gsub("Damage Item", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("jira", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("Bad-quality Item", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("Wrong Item", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("Biker Behavior", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("Fresh QUality", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("Near Expiration", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("fragile", "شکستنی", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("D:", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("Extra Item", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("gira", "", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("error", "ارور", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("c", "سی", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("B", "ب", df$Comment, ignore.case = TRUE)
df$Comment <- gsub("a4", "", df$Comment, ignore.case = TRUE)
df <- df[!(df$X == 108 | df$X == 110 | df$X == 203 | df$X == 912 | df$X == 1311), ]

english_rows <- df[grepl("a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z", df$Comment, ignore.case = TRUE), ]

short_df <- df[nchar(df$Comment) < 10, ]
df <- df[!(df$Comment == "" | df$Comment == " " | df$Comment == "  " | df$Comment == " تخم مرغ" | df$X == 63 | df$X == 219 | df$X == 942 | df$X == 1530) ,]
df <- subset(df, select = -issue_map)
df$label <- ""

# We will use this full data at the end of the project for full labeling prediction of the complete dataset
write.csv(df, "C:\\Users\\reza farshchi\\OneDrive\\Desktop\\full_data.csv", row.names=FALSE)

# Set the seed for reproducibility
set.seed(657)

# Split the data into training, validation, and test sets
train_indices <- sample(nrow(df), round(0.1*nrow(df)), replace = FALSE)
val_indices <- sample(setdiff(1:nrow(df), train_indices), round(0.45*nrow(df)), replace = FALSE)
test_indices <- setdiff(setdiff(1:nrow(df), train_indices), val_indices)

train_data <- df[train_indices, ]
val_data <- df[val_indices, ]
test_data <- df[test_indices, ]

write.csv(train_data, "C:\\Users\\reza farshchi\\OneDrive\\Desktop\\train_data.csv", row.names=FALSE)
write.csv(val_data, "C:\\Users\\reza farshchi\\OneDrive\\Desktop\\val_data.csv", row.names=FALSE)
write.csv(test_data, "C:\\Users\\reza farshchi\\OneDrive\\Desktop\\test_data.csv", row.names=FALSE)