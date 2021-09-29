# Voice-Conversion
Modify speech spoken by source speaker to give an impression that it was spoken by target speaker

We can achieve voice conversion in two ways:
a. end-to-end voice conversion
b. a combination of speech recognition and speech synthesis
In this project, only part (b), i.e., combination of speech recognition and speech synthesis has been implemented.

End-to-end voice conversion uses a single system to convert the source voice to target voice. End-to-end voice conversion, in general, requires parallel training data (same sentences spoken in source and target voices). It also requires some more techniques such as Dynamic Time Warping in order to align the similar frames of the source and target.

For the second type of method, we can have one network (Net1) which converts the speech into phonemes (speech recognition) and another network (Net2) converts the phonemes into the target voice (speech synthesis). This method does not require parallel training data, so it does not impose such restrictions on the dataset. Here, the training becomes easy as the two networks can be trained and evaluated independently.

![image](https://user-images.githubusercontent.com/79351706/135359330-d132a44b-ac0f-4568-9075-d1ba618d6209.png)

The architecture can be divided into two stages. The first stage (Net1) can be used to convert MFCCs (Mel Frequency Cepstral Coefficients) extracted from the source waveform to phonemes. These can then be fed into the next neural network (Net2) which converts phonemes to the target waveform.

![image](https://user-images.githubusercontent.com/79351706/135359361-60a289a4-98cb-4292-b60d-125d1ca8a556.png)

![image](https://user-images.githubusercontent.com/79351706/135359396-d5007276-ee7a-458c-9683-6eedd9a1ef36.png)

We can construct the target speaker wav file from the magnitude waveform obtained as output of Net2. Since we only have the magnitude spectrum and no information about the phase, this reconstruction will not be perfect. For this, we can use the Griffin-Lim algorithm.

**Datasets:** [TIMIT dataset](https://deepai.org/dataset/timit) and [CMU ARCTIC dataset](http://www.festvox.org/cmu_arctic/). TIMIT dataset contains 630 speakers’ utterances and corresponding phones along with their start and end times in the audio. This dataset has already a test and train folder with 1680 and 4620 files (audio as well as phoneme) respectively. It is used for training and testing Net1. CMU ARCTIC speech synthesis databases consist of 1132 utterances (at sampling frequency 16 kHz) per speaker spoken by different speakers (male as well as female). I have chosen one of the speakers and divided the files into 80 % training data (905 audio and 905 phoneme files) and 20 % testing data (227 audio and 227 phoneme files). These have been used for training and testing Net2. The two models are trained and tested separately. Finally, a random test sentence (source audio) from TIMIT dataset has been fed to Net1 to predict phonemes which is then be fed to Net2 to get output (target audio).

Many-to-one Voice Conversion: Net1 is trained using multiple speaker’s audio files (source) and their corresponding phones (TIMIT dataset). Net2 is trained using a single speaker audio files (target) and their corresponding phones (CMU ARCTIC dataset). 

One-to-one Voice Conversion: Net1 is trained using single speaker audio files (source) and their corresponding phones (one of the speaker of CMU ARCTIC dataset). Net2 is trained using a single speaker audio files (target) and their corresponding phones (another speaker of CMU ARCTIC dataset). 
