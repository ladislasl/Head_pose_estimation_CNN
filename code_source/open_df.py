import pyarrow.parquet as pq
import pandas as pd

df = pd.read_parquet('series.pq', engine='pyarrow')

#df = pq.read_table('series.pq').to_pandas()

first_column = df.iloc[:, 1]

print(first_column)


print(df.columns)
print(len(df.columns))