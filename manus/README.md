# Code to convert the real dataset

## Compute fingertips
### Load mvnx data to excel
Load the data to excel like explained [here](https://tutorial.xsens.com/video/importing-mvnx-into-excel/). 
You should have the following sheets:
* fingerTrackingSegmentsLeft
* fingerTrackingSegmentsRight
* positionFingersLeft
* positionFingersRight
* orientationFingersLeft
* orientationFingersRight


This can also be seen in ```recorded_data.xlsx```.


### Run ``prepare_data.py``
This will generate the target left and right points for the MANO model.