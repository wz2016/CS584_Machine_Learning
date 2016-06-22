import numpy as np
import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.xlabel('recall');
plt.ylabel('precision');
plt.title('precision-recall curve')
# plt.axis([0.8, 1.05, 0.9, 1.05])
precision_1 = np.array([
	0.485714285714
,	0.653846153846
,	0.882352941176
,	0.915492957746
,0.90410958904
,	0.881578947368


,0.971830985915
])
recall_1 = np.array([
	0.728571428571
,	0.728571428571
,	0.857142857143
,	0.928571428571
, 0.942857142857
,	0.957142857143


, 0.985714285714

])

precision_2 = np.array([
	0.457142857143
,	0.693548387097
,	0.953125
,	0.861111111111

,0.942857142857

,0.927536231884
,0.985507246377

])
recall_2 = np.array([
	0.228571428571
,	0.614285714286
,	0.871428571429
,	0.885714285714

, 0.9
, 0.914285714286
, 0.971428571429
])
plt.plot(recall_1,precision_1,'r',recall_2, precision_2, 'g');
plt.show()
