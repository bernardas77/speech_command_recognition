**Speech commands recognition**
The project classifies one word speech commands and compares between different input feature representations and models.

**Dataset**
Google's "Mini Speech Commands" dataset is used for this project, which contains 8 thousand one second clips of different speakers saying one of the commands: yes, up, stop, right, no, left, go, down. 
Each class contains a thousand samples at 16kHz. The dataset is split 80%/10%/10% between training, validation and test sets.

**Features**
These are the features used to train the models: raw waveform, spectrogram, log-Mel spectrogram, MFCC.

**Models**
model1: 2D CNN on the spectrogram
model2: 2D CNN on log-Mel
model3: 1D CNN on the raw waveform
model4: Random Forest on MFCC
model5: 2-layer GRU on MFCC
model6: CRNN (CNN + BiGRU) on log-Mel

**Results**
Below is a table with different metrics (accuracy, precision, recall, specificity) evaluated on a test set
<table>
  <thead>
    <tr>
      <th rowspan="2">modelis</th>
      <th colspan="3">tikslumas</th>
      <th colspan="3">preciziškumas</th>
      <th colspan="3">jautrumas</th>
      <th colspan="3">specifiškumas</th>
    </tr>
    <tr>
      <th>train</th><th>val</th><th>test</th>
      <th>train</th><th>val</th><th>test</th>
      <th>train</th><th>val</th><th>test</th>
      <th>train</th><th>val</th><th>test</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>model1_cnn_spectrogram</td><td>0.9636</td><td>0.8562</td><td>0.8588</td><td>0.9643</td><td>0.8599</td><td>0.8601</td><td>0.9637</td><td>0.8590</td><td>0.8582</td><td>0.9948</td><td>0.9795</td><td>0.9799</td></tr>
    <tr><td>model2_cnn_logmel</td><td>0.9584</td><td>0.8700</td><td>0.8700</td><td>0.9591</td><td>0.8737</td><td>0.8741</td><td>0.9585</td><td>0.8709</td><td>0.8690</td><td>0.9941</td><td>0.9814</td><td>0.9814</td></tr>
    <tr><td>model3_cnn1d_raw</td><td>0.6502</td><td>0.6325</td><td>0.6675</td><td>0.6514</td><td>0.6382</td><td>0.6702</td><td>0.6501</td><td>0.6383</td><td>0.6631</td><td>0.9500</td><td>0.9474</td><td>0.9526</td></tr>
    <tr><td>model4_rf_mfcc</td><td>1.0000</td><td>0.7238</td><td>0.7525</td><td>1.0000</td><td>0.7283</td><td>0.7594</td><td>1.0000</td><td>0.7281</td><td>0.7500</td><td>1.0000</td><td>0.9606</td><td>0.9646</td></tr>
    <tr><td>model5_gru_mfcc</td><td>0.9753</td><td>0.9188</td><td>0.9438</td><td>0.9757</td><td>0.9196</td><td>0.9445</td><td>0.9754</td><td>0.9196</td><td>0.9452</td><td>0.9965</td><td>0.9884</td><td>0.9920</td></tr>
    <tr><td>model6_crnn_logmel</td><td>0.9694</td><td>0.9162</td><td>0.9288</td><td>0.9703</td><td>0.9175</td><td>0.9327</td><td>0.9694</td><td>0.9169</td><td>0.9279</td><td>0.9956</td><td>0.9881</td><td>0.9898</td></tr>
  </tbody>
</table>

From the results it is apparent, that GRU and CRNN models performed the best (GRU MFCC 0.944, CRNN 0.929 accuracy), Random Forest clearly overfits the training set and 1D CNN on raw waveform performed 
the worst (0.668 accuracy).

**Conclusions**
GRU model performs the best because it models frequency changes overtime better than just a regular CNN, which is important for such input as sound.
Since Random Forest overfits the training set, grid search for optimal hyperparameter selection should be implemented in the future.
1D CNN on raw waveform performed the worst because it captures noise and doesn't capture all the composing frequencies of a sound.

**To run**
You can open the notebook in Google Colab and run it. Change runtime device from CPU to available GPU to run faster. If the notebook is run locally, first install the required dependencies.
