# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:28:02 2020

@author: Web
"""

# import pandas as pd
# df1 = pd.read_clipboard(sep=',')
# df2 = pd.read_clipboard(sep=',')
import pandas as pd
df1=pd.DataFrame(columns=list('abcde'),index = range(10))
df1.iloc[:,:]=0
df2=pd.DataFrame(columns = list('abcdef'),index = range(2,22,2))
df2.iloc[:,:]=1
print(df1)
print(df2)
df2.update(df1)
print(df2)