import pandas as pd
import os

path = r"c:\8th sem project\python-engine\data\student_performance.csv"
df = pd.read_csv(path)

print(f"Shape: {df.shape}")
print(f"File size: {os.path.getsize(path) / 1024:.0f} KB")

print("\n--- Correlation with final_score ---")
corr = df.select_dtypes(include="number").corr()["final_score"].sort_values(ascending=False)
for col, val in corr.items():
    print(f"  {col:30s} {val:+.3f}")

print("\n--- Null values ---")
print(f"  Total nulls: {df.isnull().sum().sum()}")
