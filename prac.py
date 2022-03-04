import pandas as pd
import numpy as np

s1 = pd.Series(np.random.randint(0, 10, 8))
s2 = pd.Series(np.random.randint(10, 20, 8))
s1.name = 'S_1'
s2.name = 'S_2'

df = pd.DataFrame({s1.name: s1, s2.name: s2})

# 단순 for 문 -> column name 들을 순회
for column_name in df:
    print(column_name)

# enumerate for 문 -> column 순서를 index로 하고 column name과 함께 순회
for i, column_name in enumerate(df):
    print(i, column_name)

# df.index 를 통해 df의 index들에 접근
for i in df.index:
    print(i, df[s1.name][i], df[s2.name][i])

# itertuples 을 이용하여 row를 이름있는 tuple로 변환하여 접근
for row in df.itertuples(name='Name_1'):
    print(row)

# iterrows 메서드를 사용
for row in df.iterrows():
    print(row)





