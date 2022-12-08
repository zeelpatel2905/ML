import pandas as pd

file = pd.read_csv(
    'G:/My Drive/SEM - 9 (2020)/907 - Machine Learning/Practical/task2.csv')

df = pd.DataFrame(file)

#1
print("Date:\n", df)
#2
print("Total number of rows is", df.shape[0],"& column is",df.shape[1])
#3
print(df.head(5))
#4

#5
print(df[["City"]])
#6
print(df.head(1)[["City"]])
#7
print(df.iloc[0])
#8
print(df.iloc[:,0])
#9
print(df.iloc[[1,2],0])
#10
print(df.iloc[[1,3],0])
#11
print(df.iloc[-3:])
#12