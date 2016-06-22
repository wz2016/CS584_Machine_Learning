import numpy as np

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

def getArea(precistion, recall):
	num = precistion.size-1
	Area = 0;
	for i in range(0, num):
		Area += (recall[i+1]-recall[i])* (precistion[i]+precistion[i+1])*(0.5)

	print Area
getArea(precision_1, recall_1)
getArea(precision_2, recall_2)