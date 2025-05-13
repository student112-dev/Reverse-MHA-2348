# Step 1: Importing the Modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import tabula

# Step 2 : Load the Data 
# Pdf path and Read 
pdf_path = "/workspaces/Reverse-MHA-2348/HIRAD_Loans_Database.pdf"
# Extract all tables from PDF
pdf_tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)

# Print number of tables extracted
print(f"Total tables extracted: {len(pdf_tables)}")
pdf_df = pdf_tables[0]
print("\nExtracted PDF Table Preview:")
print(pdf_df.head())

# Excel Path and Read
excel_path = "/workspaces/Reverse-MHA-2348/HIRAD_Loans_Database.xlsx"
excel_df = pd.read_excel(excel_path, sheet_name='Table 1')

print("\nExcel Table Preview:")
print(excel_df.head())


#  STEP 3: Merge Datasets
merged_df = pd.merge(excel_df, pdf_df, on='Loan_ID', how='inner')

print("\nMerged Dataset Preview:")
print(merged_df.head())

merged_df.to_csv("merged_hirad_dataset.csv", index=False)
print("✅ Merged dataset saved as 'merged_hirad_dataset.csv'")

# STEP 4: Preprocessing & Cleaning
# Drop duplicate '_y' columns
columns_to_keep = [col for col in merged_df.columns if col.endswith('_x') or col == 'Loan_ID']
df = merged_df[columns_to_keep]

# Rename columns (remove '_x' suffix)
df.columns = [col.replace('_x', '') if col != 'Loan_ID' else col for col in df.columns]

# Preview cleaned DataFrame
print("✅ Cleaned Columns:")
print(df.columns.tolist())

print("\n🔍 Data Preview:")
print(df.head())


# Drop duplicates
df.drop_duplicates(inplace=True)

# Check missing values
print("\n📉 Missing Values:")
print(df.isnull().sum())

# Drop rows with missing values (or use imputation if needed)
df.dropna(inplace=True)

# Confirm shape
print(f"\n✅ Final Shape After Cleaning: {df.shape}")

# STEP 5 : Exploratory Data Analysis (EDA)

# Distribution of Loan Status and Save as PNG
plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.tight_layout()

# Save the image
plt.savefig("loan_status_distribution.png", dpi=300)

# Show the plot
plt.show()

# Gender vs Loan Approval
plt.figure(figsize=(6, 4))
sns.countplot(x="Gender", hue="Loan_Status", data=df, palette="magma")
plt.title("Loan Status by Gender")
plt.show()
# Save the image
plt.savefig("Gender-wise_Loan_Status.png", dpi=300)

# Show the plot
plt.show()


# Credit History vs Loan Approval
plt.figure(figsize=(6,4))
sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
plt.title("Loan Status by Credit History")
plt.show() 

# Save the image
plt.savefig("Loan_Status_by_Credit_History.png", dpi=300)

# Show the plot
plt.show()

# Income vs Loan Amount
plt.figure(figsize=(6,4))
sns.scatterplot(x='ApplicantIncome', y='LoanAmount', hue='Loan_Status', data=df)
plt.title("Applicant Income vs Loan Amount")
plt.show()

# Save the image
plt.savefig("Applicant_Income_vs_Loan_amount.png", dpi=300)

# Show the plot
plt.show()

# Countplot
plt.figure(figsize=(7,5))
sns.countplot(x='Property_Area', hue='Loan_Status', data=df, palette='Set2')
plt.title("Loan Approval by Property Area")
plt.xlabel("Property Area")
plt.ylabel("Number of Applications")
plt.legend(title="Loan Status")
plt.tight_layout()
plt.show()

# Save the image
plt.savefig("Loan_Approval_by_Property_Area.png", dpi=300)

# Show the plot
plt.show()


# Saved the cleaned dataset 
df.to_csv("cleaned_loans_data.csv", index=False)
