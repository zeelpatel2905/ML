import pandas as pd

a = [1,3,7]

myvar=pd.Series(a)

print(myvar)
print(myvar[2])

b = { "day1":50,"day2":100,"day3":150}

myvar1=pd.Series(b)

print(myvar1)
print(myvar1["day3"])

c={
   "id":[1,2,3],
   "name":["a","b","c"]
   }

myvar3=pd.DataFrame(c)
print(myvar3)

d=pd.read_csv(C:/Users/zeelp/Desktop/data.csv)

myvar4=pd.DataFrame(d)

print(myvar4)