# MARS-OPEN-PROJECT

## PROJECT DESCRIPTION
We use the dataset provided which contains audio recordings of actors speaking in different emotional tones (like happy, sad, angry, etc.). For extracting features we use Whisper, a powerful pre-trained model from OpenAI to generate deep audio embeddings.These embeddings are then classified into one of 8 emotions using an MLP.The model is trained in PyTorch, evaluated using accuracy F1 Scores and confusion matrix.

## PREPROCESSING METHODOLOGY
The preprocessing pipeline begins by loading each .wav audio file from the dataset using the torchaudio library. Since some audio recordings contain multiple channels (stereo), they are converted to mono to standardize the input format. Following this, all audio waveforms are resampled to 16 kHz, which is the required input sampling rate for OpenAI's Whisper model.Once standardized, the audio is passed through the Whisper processor from HuggingFace. This processor converts raw audio into a format suitable for Whisper’s encoder, producing dense input features. These features are then forwarded through Whisper’s encoder to obtain the last hidden state, which is a sequence of deep audio embeddings.These deep audio embeddings are of vaying lengths so in order to standardize them we use afixed sizerepresentation suitable for classification by performing mean pooiling across time dimension. This results in a 512 dimensional embedding vector for each audio sample.For making the labels we will be extracting it from the file name as it is in the RAVDESS naming format. wwhich contains the emotion namein encoded for in the filename.The seed_all() function ensures reproducibility in the code, meaning you’ll get the same results every time you run your code.

## MODEL DETAILS
The primary model used is an improved Multi-Layer Perceptron (MLP). This MLP includes several layers with normalization, dropout, and non-linear activation functions to enhance generalization and avoid overfitting. It progressively reduces the embedding dimension through dense layers and outputs a probability distribution over eight emotion classes(neutral,calm,happy,sad,angry,digusted,fearful,surprised)

### TRAINING DETAILS
Training was performed on 80% of the dataset.The model was trained using the cross entropy loss functioning with label smoothing of 0.1 and class weights to handle class imbalance.Optimizer used was AdamW with lr of 1e-4 and weight decay regularization of 1e-4. I also used an lr scheduler (ReduceLROnPlateau) withpatience of 3 epochs which reduces the learning rate by a factor of 0.5 if validation performance doesnt increase for 3 epochs. The model was trained for 90 epochs

## VALIDATION METRICS
Validation was performed on 20% of the total data.The validation accuracy was 88.59% and F1 score was 88.96%.
Confusion Matrix was:
[[35  1  2  0  0  0  0  0]
 [ 1 69  1  3  0  0  1  0]
 [ 4  1 63  1  1  2  2  1]
 [ 2  8  0 60  0  5  0  0]
 [ 0  0  0  1 71  0  2  1]
 [ 0  1  1  9  2 62  0  0]
 [ 0  0  1  0  0  0 36  2]
 [ 0  0  0  0  0  0  0 39]]
 #### Per Class Accuracy
 Class Neutral Accuracy: 0.9211
Class Calm Accuracy: 0.9200
Class Happy Accuracy: 0.8400
Class Sad Accuracy: 0.8000
Class Angry Accuracy: 0.9467
Class Fearful Accuracy: 0.8267
Class Disgust Accuracy: 0.9231
Class Surprised Accuracy: 1.0000




