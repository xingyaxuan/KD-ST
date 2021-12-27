# KD-ST


This is a Pytorch implementation of KD-ST: Distillation Knowledge-Based Space-Time Data Prediction on Industrial IoT Edge Devices.

Currently, IIoT tends to offer significant efficiency and productivity gains to industrial operations. 
Compared to traditional IoT consumer-oriented devices, IIoT is large-scale and faces more life-threatening or high-risk situations due to system failures and downtime. 
IIoT imperatively requires energy-saving equipment and secure operation to overcome the existences of high energy consumption, 
limited battery capacity, substantial safety hazard, and complex data processing. 
Outstandingly, edge computing has emerged as a promising technology to support IIoT systems by allocating computation and storage resources at the network edge. 
The proposed KD-ST can efficiently slove the problem that the existing deep learning algorithms are too high complexity to deploy on edge devices, 
achieved a good tradeoff between execution cost and model accuracy in model inference.


# The Code
## Requirements:
* pytorch
* scipy
* numpy
* matplotlib
* pandas
* math

## Implement
You can get the dataset at https://www.kaggle.com/javi2270784/gas-sensor-array-temperature-modulation.

Our teacher model is stored in model <model.pt>.
You can run the following four files：
1. LSTM-based student network <KD_ST_LSTM.py>
2. Student network based on 1-dimensional convolution <KD_ST_1DCNN.py>
3. LSTM-based transfer student network <Transfer_LSTM.py>
4. The transfer student network based on 1-dimensional convolution <Transfer_1DCNN.py>
