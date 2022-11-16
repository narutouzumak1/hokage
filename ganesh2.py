import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#find shape of data
df=pd.read_csv('/home/gaurav/Desktop//Heart.csv')
df.shape
df.notnull()
df.isin([0]).any().any()
(df==0).sum()
df["Age"].mean()
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("Heart.csv")
df.head()
cdf = df[['Age','MaxHR','Sex','RestECG','Chol','RestBP']]
cdf.head(9)
plt.scatter(cdf.Age, cdf.MaxHR,  color='blue')
plt.xlabel("Age")
plt.ylabel("MaxHR")
plt.show()
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
plt.scatter(train.Chol, train.MaxHR,  color='blue')
plt.xlabel("Chol")
plt.ylabel("MaxHR")
plt.show()
result = df.dtypes
print(result)
