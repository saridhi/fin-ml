import pandas as pd
from barclaysdata import BarclaysData
from functools import reduce, lru_cache
import pdb
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import collections
from sklearn.decomposition import PCA

class GenericAlgo(object):
 
 def __init__(self):
     self.data_dict = collections.OrderedDict()
 
     #Only allows Weekly conversion for now
     #period = 'D' for daily and 'W' for weekly
 def convert_data(self, data, period):
     weekly_conv = 'W-FRI'
     size = len(data.columns)
     #pdb.set_trace()
     if period=='D':
     return data
     elif size == 1:
     return data.resample(weekly_conv).last()
     elif size == 4:
     op = data.iloc[:,0].resample(weekly_conv).first()
     hi = data.iloc[:,1].resample(weekly_conv).max()
     lo = data.iloc[:,2].resample(weekly_conv).min()
     cl = data.iloc[:,3].resample(weekly_conv).last()
     return pd.concat([op, hi, lo, cl], axis=1)
     
 # Data can be retrieved as a list or a pd.Dataframe of Series
 @lru_cache(maxsize=128)
 def get_data(self, start_date, end_date, merged=False, period='D'): 
     print('Retrieving data...')
     for k,v in self.tickers.items():
     lookup_method = "get_"+v[0]+"_data"
     data=getattr(self, lookup_method)(k,v, start_date, end_date)
     #pdb.set_trace()
     data = data.fillna(method='ffill', axis=0)
     data = self.convert_data(data, period)
     self.data_dict[k] = data
     print('Data Retrieved.')
     return_data = self.data_dict if not(merged) else self.merge_data(self.data_dict)
     return return_data
 
 #Standard scaling 
 #Optional pre_scaled argument can provide a scaler
 def scale(self, x):
     if self.cols_to_scale == None:
     return x
     x_other = x.iloc[:,list(set(list(range(0, len(x.columns))))-set(self.cols_to_scale))]
     x = x.iloc[:,self.cols_to_scale]
     if self.scaler == None:
     scaler = StandardScaler().fit(x)
     self.scaler = scaler
     else:
     scaler = self.scaler
     scaled_x = scaler.transform(x)
     scaled_x = pd.DataFrame(scaled_x, columns = x.columns, index = x.index)
     x = scaled_x.join(x_other, how='outer')
     return x
     
 #Standard pca 
 def pca(self, x, components=None):
     if self.cols_to_pca == None:
     return x
     x_other = x.iloc[:,list(set(list(range(0, len(x.columns))))-set(self.cols_to_pca))]
     x = x.iloc[:,self.cols_to_pca]
     scaled_x = StandardScaler().fit_transform(x)
     if self.pcaler == None:
     if (components==None):
     components = scaled_x.shape[1]
     pcaler = PCA(n_components=components)
     self.pcaler = pcaler
     else:
     pcaler = self.pcaler
     pca_x = pcaler.fit_transform(scaled_x)
     scaled_x = pd.DataFrame(pca_x, columns = x.columns, index = x.index)
     x = scaled_x.join(x_other, how='outer')
     return x
     
 #Rolling scaling to ensure out of sample testing
 #TODO
# def roll_scale(self, x_test, x):
# if self.cols_to_scale == None:
# return x_test
# x_other = x.iloc[:,list(set(list(range(0, len(x.columns))))-set(self.cols_to_scale))]
# x = x.iloc[:,self.cols_to_scale]
# x_test = x_test.iloc[:,self.cols_to_scale]
# x_all = x.append(x_test)
# scaled_x = x_all.expanding(min_periods=len(x)).apply(lambda x: self.scale(pd.DataFrame(x).iloc[len(x)-1:,], pd.DataFrame(x)[:len(x)-1]))
# scaled_x = pd.DataFrame(scaled_x, columns = x_all.columns, index = x_all.index)
# x = scaled_x.join(x_other, how='outer')
# return x
 
 #Merge data_dict into one dataframe
 def merge_data(self, data_dict):
     data = data_dict.values()
     temp_data = reduce(lambda x,y: x.join(y, how='outer'), data)
     temp_data = temp_data.fillna(method='pad')
     self.data_df = temp_data.dropna(axis=0)
     #self.data_df.sort_index(inplace=True)
     self.data_df.columns = list(data_dict.keys())
     return self.data_df
 
 #Ensure data is in a pd.Dataframe before splitting in train/test
 #If threshold given, use that, else use the train/test cutoff points with optional scaling
 #Returns y_train, y_test, x_train, x_test as DataFrames
 def get_traintest(self, x, y, threshold = 0.8):
     #pdb.set_trace()
     common_ind = x.index.intersection(y.index)
     y = y.loc[common_ind]
     x = x.loc[common_ind]
     merged = y.to_frame().join(x)
     merged = merged.dropna(how='all')
     y = merged.iloc[:,0]
     x = merged.iloc[:,1:]
     #pdb.set_trace()
     cutoff = int(np.floor(len(merged)*threshold))
     end_cut = len(merged)
     y_train = y.iloc[0:cutoff]
     y_test = y.iloc[cutoff+1:end_cut]
     x_train = x.iloc[0:cutoff, :]
     x_test = x.iloc[cutoff+1:end_cut,:]
     return y_train.to_frame().values.ravel(), y_test.to_frame().values.ravel(), x_train, x_test