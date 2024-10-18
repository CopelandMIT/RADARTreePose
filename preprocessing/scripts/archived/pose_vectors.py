import pandas as pd
import csv

# Import CSV file and store each column as a separate list of strings
with open('pose_vectors.csv') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Get header row
    data = [list(row) for row in reader]

# Create a Pandas dataframe from the lists of strings
pose_df = pd.DataFrame(data, columns=header)
print(pose_df.columns)
# Print the dataframe
# print(pose_df[["CRW2L"]])
