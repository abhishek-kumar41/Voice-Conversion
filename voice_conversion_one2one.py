import glob
import librosa
import IPython.display as ipd
import numpy as np
from scipy import signal

win_length = 0.025
hop_length = 0.005

arctic_wav_data_path = 'cmu_us_bdl_arctic/wav/arctic_*.wav'
arctic_wav = glob.glob(arctic_wav_data_path)
arctic_wav.sort()
print(len(arctic_wav))
num_arctic_train  = int(0.8*len(arctic_wav))
# print(num_arctic_train)
num_arctic_test = len(arctic_wav) - num_arctic_train
# print(num_arctic_test)
# print(arctic_wav)
arctic_train_wav = arctic_wav[:num_arctic_train]
print(len(arctic_train_wav))
arctic_test_wav = arctic_wav[num_arctic_train:len(arctic_wav)]
print(len(arctic_test_wav))

arctic_phns_data_path = 'cmu_us_bdl_arctic/lab/*.lab'
arctic_phns = glob.glob(arctic_phns_data_path)
arctic_phns.sort()
print(len(arctic_phns))
# phns =  open(arctic_phns[0], "r").read()
# print(phns)
num_arctic_train_phns  = int(0.8*len(arctic_phns))
# print(num_arctic_train_phns)
num_arctic_test_phns = len(arctic_phns) - num_arctic_train_phns
# print(num_arctic_test_phns)
arctic_train_phns = arctic_phns[:num_arctic_train_phns]
print(len(arctic_train_phns))
arctic_test_phns = arctic_phns[num_arctic_train_phns:len(arctic_phns)]
print(len(arctic_test_phns))

arctic_wav_data_path2 = 'cmu_us_slt_arctic/wav/arctic_*.wav'
arctic_wav2 = glob.glob(arctic_wav_data_path2)
arctic_wav2.sort()
print(len(arctic_wav2))
num_arctic_train2  = int(0.8*len(arctic_wav2))
# print(num_arctic_train2)
num_arctic_test2 = len(arctic_wav2) - num_arctic_train2
# print(num_arctic_test2)
# print(arctic_wav2)
arctic_train_wav2 = arctic_wav2[:num_arctic_train2]
print(len(arctic_train_wav2))
arctic_test_wav2 = arctic_wav2[num_arctic_train2:len(arctic_wav2)]
print(len(arctic_test_wav2))
# print(arctic_test_wav2)

arctic_phns_data_path2 = 'cmu_us_slt_arctic/lab/*.lab'
arctic_phns2 = glob.glob(arctic_phns_data_path2)
arctic_phns2.sort()
print(len(arctic_phns2))
# phns =  open(arctic_phns2[0], "r").read()
# print(phns)
num_arctic_train_phns2  = int(0.8*len(arctic_phns2))
# print(num_arctic_train_phns2)
num_arctic_test_phns2 = len(arctic_phns2) - num_arctic_train_phns2
# print(num_arctic_test_phns2)
arctic_train_phns2 = arctic_phns2[:num_arctic_train_phns2]
print(len(arctic_train_phns2))
arctic_test_phns2 = arctic_phns2[num_arctic_train_phns2:len(arctic_phns2)]
print(len(arctic_test_phns2))

phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
print(len(phns))


def load_vocab():
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn

phn2idx, idx2phn = load_vocab()

print(idx2phn)
print(phn2idx['pau'])


def string_to_matrix_dict(string):
    line_split = list(string.split("\n"))
    matrix = []

    for item in line_split:
        line = []
        for data in item.split(" "):
            line.append(data)

        matrix.append(line)
    return matrix[0:len(matrix)-1]


def get_all_feature_phoneme(arctic_train_wav, arctic_train_phns):
    from tqdm import tqdm
    train1_mfccs = []
    train1_phns = []
    max_duration=4

    for i in tqdm(range(len(arctic_train_wav))):

        time_step1_mfccs=[]
        time_step1_phns=[]
        y, sr = librosa.load(arctic_train_wav[i], sr=None)
        phoneme =  open(arctic_train_phns[i], "r").read()
        phoneme = string_to_matrix_dict(phoneme)

        if(len(y) > sr*max_duration):
            y=y[:sr*max_duration]
        else:
            y=np.pad(y, (0, sr*max_duration-len(y)), 'constant')

        win = int(win_length*sr)
        hop = int(hop_length*sr)

        y_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, win_length=win, hop_length=hop)

        count = 0

        for j in range(0,len(y),hop):

            count = count+1
            index = int(j/hop)

            time_step1_mfccs.append(y_mfcc[:,index])
            x=0
            for k in range(1,len(phoneme)-1):
                start_index = int(sr*(float(phoneme[k][0])))
                next_index = int(sr*(float(phoneme[k+1][0])))
                if(j>=start_index and j<=next_index):
                    phn_str = phoneme[k+1][2]
                    phn_label = phn2idx[phn_str]
                    if(phn_label==44):
                        phn_label=0
                    phn_one_hot = np.eye(len(phns))[phn_label]
                    time_step1_phns.append(phn_one_hot)
    #                 time_step1_phns.append(phn_label)
                    x=x+1
                    break
            if(x==0):
                phn_label = 0
                phn_one_hot = np.eye(len(phns))[phn_label]
                time_step1_phns.append(phn_one_hot) 
    #             time_step1_phns.append(phn_label)
        train1_mfccs.append(np.array(time_step1_mfccs))
        train1_phns.append(np.array(time_step1_phns))
    train1_mfccs=np.array(train1_mfccs)
    train1_phns=np.array(train1_phns)

    return train1_mfccs, train1_phns


def get_one_feature_phoneme(arctic_train_wav, arctic_train_phns, sample_no):
    from tqdm import tqdm
    train1_mfccs = []
    train1_phns = []
    max_duration=4

    for i in tqdm(range(sample_no, sample_no+1)):

        time_step1_mfccs=[]
        time_step1_phns=[]
        y, sr = librosa.load(arctic_train_wav[i], sr=None)
        phoneme =  open(arctic_train_phns[i], "r").read()
        phoneme = string_to_matrix_dict(phoneme)

        if(len(y) > sr*max_duration):
            y=y[:sr*max_duration]
        else:
            y=np.pad(y, (0, sr*max_duration-len(y)), 'constant')

        win = int(win_length*sr)
        hop = int(hop_length*sr)

        y_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, win_length=win, hop_length=hop)

        count = 0

        for j in range(0,len(y),hop):

            count = count+1
            index = int(j/hop)

            time_step1_mfccs.append(y_mfcc[:,index])
            x=0
            for k in range(1,len(phoneme)-1):
                start_index = int(sr*(float(phoneme[k][0])))
                next_index = int(sr*(float(phoneme[k+1][0])))
                if(j>=start_index and j<=next_index):
                    phn_str = phoneme[k+1][2]
                    phn_label = phn2idx[phn_str]
                    if(phn_label==44):
                        phn_label=0
                    phn_one_hot = np.eye(len(phns))[phn_label]
                    time_step1_phns.append(phn_one_hot)
    #                 time_step1_phns.append(phn_label)
                    x=x+1
                    break
            if(x==0):
                phn_label = 0
                phn_one_hot = np.eye(len(phns))[phn_label]
                time_step1_phns.append(phn_one_hot) 
    #             time_step1_phns.append(phn_label)
        train1_mfccs.append(np.array(time_step1_mfccs))
        train1_phns.append(np.array(time_step1_phns))
    train1_mfccs=np.array(train1_mfccs)
    train1_phns=np.array(train1_phns)

    return train1_mfccs, train1_phns


train1_mfccs, train1_phns = get_all_feature_phoneme(arctic_train_wav, arctic_train_phns)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True), input_shape=(800,13)))
model.add(layers.Dropout(0.1))
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True), input_shape=(800,128)))
model.add(layers.Dropout(0.1))
model.add(layers.TimeDistributed(layers.Dense(64, activation="tanh")))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation="tanh"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation="tanh"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(61, activation="softmax"))
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

BATCH_SIZE=32
EPOCHS=5

history = model.fit(np.array(train1_mfccs), np.array(train1_phns), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

test1_mfccs, test1_phns = get_all_feature_phoneme(arctic_test_wav, arctic_test_phns)

pred_cat = model.predict(np.array(test1_mfccs))
pred = np.argmax(pred_cat, axis=-1)
# print(pred)

y_te_true = np.argmax(np.array(test1_phns), -1)

print(pred.shape)
# print(y_te_true)
print(np.array(y_te_true).shape)

# print(len(pred))
# pred=pred.T
# y_te_true=y_te_true.T

height,width=pred.shape
print(height)
print(width)

acc_count=0
for i in range(height):
    for j in range(width):
        if(pred[i,j] == y_te_true[i,j]):
#             if(pred[i,j]!=0):
                acc_count = acc_count+1
accuracy=acc_count/(height*width)
print(f"Accuracy is {accuracy}")

# Take a random sample from test data: 0 to 226
sample_no = 220
test_feature, test_phns=get_one_feature_phoneme(arctic_test_wav, arctic_test_phns, sample_no)

pred_cat = model.predict(np.array(test_feature))
pred = np.argmax(pred_cat, axis=-1)
# print(pred)

y_te_true = np.argmax(np.array(test_phns), -1)

print(pred.shape)
# print(y_te_true)
print(np.array(y_te_true).shape)

# print(len(pred))

# pred=pred.T
# y_te_true=y_te_true.T

height,width=pred.shape
print(height)
print(width)

acc_count=0
for i in range(height):
    for j in range(width):
        if(pred[i,j] == y_te_true[i,j]):
#             if(pred[i,j]!=0):
                acc_count = acc_count+1
accuracy=acc_count/(height*width)
print(f"Accuracy is {accuracy}")

print(pred)
print(y_te_true)

print(pred_cat.shape)


def get_all_mel_phoneme(arctic_train_wav2, arctic_train_phns2):
    from tqdm import tqdm
    train2_mel = []
    train2_phns = []
    max_duration=4

    for i in tqdm(range(len(arctic_train_wav2))):

        time_step2_mel=[]
        time_step2_phns=[]
        y, sr = librosa.load(arctic_train_wav2[i], sr=None)
        phoneme =  open(arctic_train_phns2[i], "r").read()
        phoneme = string_to_matrix_dict(phoneme)

        if(len(y) > sr*max_duration):
            y=y[:sr*max_duration]
        else:
            y=np.pad(y, (0, sr*max_duration-len(y)), 'constant')

        win = int(win_length*sr)
        hop = int(hop_length*sr)

        y_mel = librosa.feature.melspectrogram(y=y, sr=sr, win_length=win, hop_length=hop)

        count = 0

        for j in range(0,len(y),hop):

            count = count+1
            index = int(j/hop)

            time_step2_mel.append(y_mel[:,index])
            x=0
            for k in range(1,len(phoneme)-1):
                start_index = int(sr*(float(phoneme[k][0])))
                next_index = int(sr*(float(phoneme[k+1][0])))
                if(j>=start_index and j<=next_index):
                    phn_str = phoneme[k+1][2]
                    phn_label = phn2idx[phn_str]
                    if(phn_label==44):
                        phn_label=0
                    phn_one_hot = np.eye(len(phns))[phn_label]
                    time_step2_phns.append(phn_one_hot)
    #                 time_step2_phns.append(phn_label)
                    x=x+1
                    break
            if(x==0):
                phn_label = 0
                phn_one_hot = np.eye(len(phns))[phn_label]
                time_step2_phns.append(phn_one_hot) 
    #             time_step2_phns.append(phn_label)
        train2_mel.append(np.array(time_step2_mel))
        train2_phns.append(np.array(time_step2_phns))
    train2_mel=np.array(train2_mel)
    train2_phns=np.array(train2_phns)

    return train2_mel, train2_phns


def get_one_mel_phoneme(arctic_train_wav2, arctic_train_phns2, sample_no):
    from tqdm import tqdm
    train2_mel = []
    train2_phns = []
    max_duration=4

    for i in tqdm(range(sample_no, sample_no+1)):

        time_step2_mel=[]
        time_step2_phns=[]
        y, sr = librosa.load(arctic_train_wav2[i], sr=None)
        phoneme =  open(arctic_train_phns2[i], "r").read()
        phoneme = string_to_matrix_dict(phoneme)

        if(len(y) > sr*max_duration):
            y=y[:sr*max_duration]
        else:
            y=np.pad(y, (0, sr*max_duration-len(y)), 'constant')

        win = int(win_length*sr)
        hop = int(hop_length*sr)

        y_mel = librosa.feature.melspectrogram(y=y, sr=sr, win_length=win, hop_length=hop)

        count = 0

        for j in range(0,len(y),hop):

            count = count+1
            index = int(j/hop)

            time_step2_mel.append(y_mel[:,index])
            x=0
            for k in range(1,len(phoneme)-1):
                start_index = int(sr*(float(phoneme[k][0])))
                next_index = int(sr*(float(phoneme[k+1][0])))
                if(j>=start_index and j<=next_index):
                    phn_str = phoneme[k+1][2]
                    phn_label = phn2idx[phn_str]
                    if(phn_label==44):
                        phn_label=0
                    phn_one_hot = np.eye(len(phns))[phn_label]
                    time_step2_phns.append(phn_one_hot)
    #                 time_step2_phns.append(phn_label)
                    x=x+1
                    break
            if(x==0):
                phn_label = 0
                phn_one_hot = np.eye(len(phns))[phn_label]
                time_step2_phns.append(phn_one_hot) 
    #             time_step2_phns.append(phn_label)
        train2_mel.append(np.array(time_step2_mel))
        train2_phns.append(np.array(time_step2_phns))
    train2_mel=np.array(train2_mel)
    train2_phns=np.array(train2_phns)

    return train2_mel, train2_phns


train2_mel, train2_phns = get_all_mel_phoneme(arctic_train_wav2, arctic_train_phns2)
print(train2_mel.shape)
print(train2_phns.shape)

model = keras.Sequential()
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True), input_shape=(800,61)))
model.add(layers.Dropout(0.1))
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True), input_shape=(800,128)))
model.add(layers.Dropout(0.1))
model.add(layers.TimeDistributed(layers.Dense(64, activation="tanh")))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation="tanh"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation="tanh"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation="linear"))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()

BATCH_SIZE=64
EPOCHS=20

history=model.fit(train2_phns,train2_mel,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_split=0.1,verbose=1)

# Take a random sample from test data for Net2: 0 to 226
sample_no = 4
test2_mel, test2_phns=get_one_mel_phoneme(arctic_test_wav2, arctic_test_phns2, sample_no)

# Eval
pred_mel = model.predict(np.array(test2_phns))
#pred_mel = model.predict(np.array(pred_cat))
pred_mel=pred_mel.T
print(np.array(test2_phns).shape)
print(pred_mel.shape)
pred_mel = pred_mel[:,:,0]
print(pred_mel.shape)
# # # print(np.array(test2_mel).shape)
# # S_inv = librosa.feature.inverse.mel_to_stft(pred_mel, sr=sr)
# # y_inv = librosa.griffinlim(S_inv)
# # # ipd.Audio(y, rate=sr, autoplay=True) # load a local WAV file
sr=16000
y_inv=librosa.feature.inverse.mel_to_audio(pred_mel, sr=sr, win_length=400, hop_length=80)
print(len(y_inv))
print(len(y_inv)/sr)

import soundfile as sf
sf.write('output.wav',y_inv, sr)
