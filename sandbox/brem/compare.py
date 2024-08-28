import pandas as pd

pd.option_context(
    'display.width',None,
    'display.max_rows',None, 
    'display.max_columns',None,
    'display.float_format','{:,.2f}'.format)

df1 = pd.read_pickle("df1.pkl")
df2 = pd.read_pickle("df2.pkl")

print()
print(df1.columns)
print(df2.columns)

print()
print(df1.size)
print(df2.size)

print()
print(df1.is_e.sum())
print(df2.is_e.sum())

print()
print(df1.weight.sum())
print(df2.weight.sum())

print()
print(df1.rho.sum())
print(df2.rho.sum())

print()
print(df1.info())
print(df2.info())

print()
print(df1.describe().T)
print(df2.describe().T)

print()
print(df1.compare(df2))
