TLD
===

This is a C++ implementation of the Tracking-Learning-Detection (TLD) algorithm.

The code is functional right now, but not stable, i.e. little changes of parameters will lead the result turning worse.


### Following is the performance evaluation on TLD dataset measured by Precision / Recall / F-measure.

|Sequence      |Frames	|OpenTLD(MatLab)           | My implementation(404dd93302)
|:-----:       |:-----:	|:-----------:             | :---:
|1.David       |761 	  |**1.00 / 1.00 / 1.00**    |0.968379 / 0.968379 / 0.968379
|2.Jumping     |313 	  |**1.00 / 1.00 / 1.00**    |1.000000 / 0.990385 / 0.995169
|3.Pedestrian 1|140   	|**1.00 / 1.00 / 1.00**    |1.000000 / 0.992806 / 0.996390
|4.Pedestrian 2|338   	|**0.89 / 0.92 / 0.91**    |0.842857 / 0.445283 / 0.582716
|5.Pedestrian 3|184	    |**0.99 / 1.00 / 0.99**    |**0.987097 / 1.000000 / 0.993506**
|6.Car		     |945	    |0.92 / 0.97 / 0.94	       |**0.928339 / 0.996503 / 0.961214**
|7.Motocross	 |2665	  |0.89 / 0.77 / 0.83        |**0.821918 / 0.849858 / 0.835655**
|8.Volkswagen  |8576	  |0.80 / 0.96 / 0.87        |**0.834528 / 0.997275 / 0.908672**
|9.Carchase	   |9928	  |**0.86 / 0.70 / 0.77**    |0.798744 / 0.646189 / 0.714413
|10.Panda	     |3000	  |0.58 / 0.63 / 0.60        |**0.630420 / 0.692561 / 0.660031**
