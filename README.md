An implementation of a feature based algorithm inspired by Hyndman's work on the FFORMS.<br>
https://robjhyndman.com/papers/fforms.pdf<br><br>

The idea is to compute rolling features and use them to compute the t+h next points.<br>
The computation of the rolling features is based on pandas and the forecasting use the scikit-learn API.<br><br>
To use the FBE, one just have to specify his sklearn model, the forecast horizon, the seasonal length (used to compute rolling features) and the freq of the datas. <br>
Tests suggests that increasing the size of the seasonal length will increase performance on forecasts but also increase the performance time.
