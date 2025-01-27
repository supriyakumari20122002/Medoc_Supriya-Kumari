import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

file_path = Path.home() / 'Downloads' / 'Chrome Downloads' / 'hospital_patient_data (1).csv'
df = pd.read_csv(file_path)

print("Initial Dataset:")
print(df.to_string())

df = df.drop_duplicates()
df['Age'].fillna(df['Age'].mean(), inplace=True)

if 'Hospital Department' in df.columns:
    df['Hospital Department'] = df['Hospital Department'].str.title()

df.to_csv("cleaned_hospital_data.csv", index=False)
print("Cleaned Dataset:")
print(df.to_string())

stats = {
    "Metric": ["Age", "Length of Stay"],
    "Mean": [df["Age"].mean(), df["Length of Stay"].mean()],
    "Median": [df["Age"].median(), df["Length of Stay"].median()],
    "Standard Deviation": [df["Age"].std(), df["Length of Stay"].std()]
}

stats_df = pd.DataFrame(stats)
print("Statistical Analysis:")
print(stats_df)

avg_stay = df.groupby("Hospital Department")["Length of Stay"].mean()

plt.figure(figsize=(8, 6))
avg_stay.plot(kind="bar", color="skyblue")
plt.title("Average Length of Stay by Department")
plt.xlabel("Hospital Department")
plt.ylabel("Average Length of Stay")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df["Age"], bins=10, kde=True, color="green")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

diag_counts = df["Diagnosis"].value_counts()

plt.figure(figsize=(8, 6))
diag_counts.plot(kind="pie", autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title("Diagnosis Distribution")
plt.ylabel("")
plt.show()
