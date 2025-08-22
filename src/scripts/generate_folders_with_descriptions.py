import os
import time
import pandas as pd
from tqdm import tqdm
PROCESSED_DATA_FOLDER = "data_processed"
TREES_PATH = f"{PROCESSED_DATA_FOLDER}/trees"
DATA_FOLDER = f"{PROCESSED_DATA_FOLDER}/sources"
COMPANY_LIST_FILE_NAME = "Bay-Area-Companies-List"
COMPANY_LIST_FILE_PATH = f"{DATA_FOLDER}/{COMPANY_LIST_FILE_NAME}.csv"
DESCRIPTION_FILE_NAME = "description.txt"

FULL_COMPANY_DESCRIPTION = '''
Name: {name}
Location: {location}
Tags: {tags}
Description: {description}'''

print(f"Loading original company list from {COMPANY_LIST_FILE_PATH}")
original_df = pd.read_csv(COMPANY_LIST_FILE_PATH)
print(f"Loaded {len(original_df)} companies")
print(original_df.head())

print("Removing duplicates")
original_df = original_df.drop_duplicates(subset=["Company Name"])
print(f"Removed duplicates, {len(original_df)} companies remaining")

# resetting index
print("Resetting index")
original_df = original_df.reset_index(drop=True)

for i, row in tqdm(original_df.iterrows(), total=len(original_df)):
    company_description = FULL_COMPANY_DESCRIPTION.format(
        name=row["Company Name"],
        location=row["Location"],
        tags=row["Tags"],
        description=row["Description"]
    )
    original_df.at[i, "Company Description"] = company_description
    
    print(f"Creating folder for company '{row['Company Name']}' if it does not exist...")
    company_name_cleaned = row['Company Name'].strip().replace(' ', '_').replace('.', '_')
    company_folder = f"{TREES_PATH}/tree_{i}_{company_name_cleaned}"
    os.makedirs(company_folder, exist_ok=True)
    
    company_description_path = f"{company_folder}/{DESCRIPTION_FILE_NAME}"
    print(f"Saving company description to {company_description_path}")
    with open(company_description_path, "w") as f:
        f.write(company_description)
    print(f"Company description saved to {company_description_path}")
    # time.sleep(1) 

new_file_path = f"{PROCESSED_DATA_FOLDER}/{COMPANY_LIST_FILE_NAME}_processed.csv"
print(f"Saving processed company list to {new_file_path}")
original_df.to_csv(new_file_path, index=True)
