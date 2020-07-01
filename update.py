# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:56:45 2020

@author: Web
"""

import pandas as pd
pd.options.display.max_columns = 40
pd.options.display.max_rows = 150
pd.options.display.width = 180

lo = pd.read_pickle('archive/out_2020-06-29 12-57-11.pkl')
out = pd.read_pickle('out.pkl')

bc =  pd.read_csv('in/products-2020-06-29.csv')
bc = bc.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

lo.set_index(['Item Type','Product Code/SKU'],inplace=True)
out.set_index(['Item Type','Product Code/SKU'],inplace=True)
bc.set_index(['Item Type','Product SKU'],inplace=True)

error = lo[~lo.index.isin(bc.index)]

conf = ['Product Description',
 'Price',
 'Cost Price',
 'Retail Price',
 'Sale Price',
 'Product Warranty',
 'Allow Purchases?',
 'Product Visible?',
 'Track Inventory',
 'Current Stock Level',
 'Category',
 'Product Image File - 1',
 'Product Image Description - 1',
 'Product Image Sort - 1',
 'Product Image File - 2',
 'Product Image Description - 2',
 'Product Image Sort - 2',
 'Search Keywords',
 'Page Title',
 'Meta Keywords',
 'Meta Description',
 'Product UPC/EAN',
 'Product Image File - 3',
 'Product Image File - 4',
 'Product Image File - 5',
 'Product Image Description - 3',
 'Product Image Description - 4',
 'Product Image Description - 5',
 'Product Image Sort - 3',
 'Product Image Sort - 4',
 'Product Image Sort - 5',
 'Product Inventoried',
 'Product ID',
 'Product Weight']