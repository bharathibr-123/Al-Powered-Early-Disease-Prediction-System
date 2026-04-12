import pandas as pd

df = pd.read_csv('kidney.csv')
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Last column (target):", df.columns[-1])