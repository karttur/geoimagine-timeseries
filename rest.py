'''
Created on 9 Mar 2019

@author: thomasgumbricht
'''

from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
 
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
 
series = read_csv('/Users/thomasgumbricht/Downloads/shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

upsampled = series.resample('D', convention='end').asfreq()
'''
upsampled = series.resample('D')
series.plot()
pyplot.show()
'''
print (upsampled)

