import glob
import librosa
import IPython.display as ipd
import numpy as np
from scipy import signal

win_length = 0.025
hop_length = 0.005

timit_train_wav_data_path = 'timit/data/TRAIN/*/*/*.WAV.wav'
timit_train_wav = glob.glob(timit_train_wav_data_path)
timit_train_wav.sort()
print(f'Total source train wav files: {len(timit_train_wav)}')

timit_train_phns_data_path = 'timit/data/TRAIN/*/*/*.PHN'
timit_train_phns = glob.glob(timit_train_phns_data_path)
timit_train_phns.sort()

timit_test_wav_data_path = 'timit/data/TEST/*/*/*.WAV.wav'
timit_test_wav = glob.glob(timit_test_wav_data_path)
timit_test_wav.sort()
print(f'Total source test wav files: {len(timit_test_wav)}')

timit_test_phns_data_path = 'timit/data/TEST/*/*/*.PHN'
timit_test_phns = glob.glob(timit_test_phns_data_path)
timit_test_phns.sort()

arctic_wav_data_path2 = 'cmu_us_slt_arctic/wav/arctic_*.wav'
arctic_wav2 = glob.glob(arctic_wav_data_path2)
arctic_wav2.sort()
print(f'Total target wav files: {len(arctic_wav2)}')
num_arctic_train2  = int(0.8*len(arctic_wav2))
num_arctic_test2 = len(arctic_wav2) - num_arctic_train2
arctic_train_wav2 = arctic_wav2[:num_arctic_train2]
print(f'Total target train wav files: {len(arctic_train_wav2)}')
arctic_test_wav2 = arctic_wav2[num_arctic_train2:len(arctic_wav2)]
print(f'Total target test wav files: {len(arctic_test_wav2)}')

arctic_phns_data_path2 = 'cmu_us_slt_arctic/lab/*.lab'
arctic_phns2 = glob.glob(arctic_phns_data_path2)
arctic_phns2.sort()
num_arctic_train_phns2  = int(0.8*len(arctic_phns2))
num_arctic_test_phns2 = len(arctic_phns2) - num_arctic_train_phns2
arctic_train_phns2 = arctic_phns2[:num_arctic_train_phns2]
arctic_test_phns2 = arctic_phns2[num_arctic_train_phns2:len(arctic_phns2)]

phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
print(f'Total no of phones: {len(phns)}')


def load_vocab():
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn


phn2idx, idx2phn = load_vocab()
print(idx2phn)


def string_to_matrix_dict(string):
    line_split = list(string.split("\n"))
    matrix = []

    for item in line_split:
        line = []
        for data in item.split(" "):
            line.append(data)

        matrix.append(line)
    return matrix[0:len(matrix)-1]


def get_all_feature_phoneme(timit_train_wav, timit_train_phns):
    from tqdm import tqdm
    train1_mfccs = []
    train1_phns = []
    max_duration=4

    for i in tqdm(range(len(timit_train_wav))):

        time_step1_mfccs=[]
        time_step1_phns=[]
        y, sr = librosa.load(timit_train_wav[i], sr=None)
        phoneme =  open(timit_train_phns[i], "r").read()
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
                start_index = int(phoneme[k][0])
                end_index = int(phoneme[k][1])
                if(j>=start_index and j<=end_index):
                    phn_str = phoneme[k][2]
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


def get_one_feature_phoneme(timit_train_wav, timit_train_phns, sample_no):
    from tqdm import tqdm
    train1_mfccs = []
    train1_phns = []
    max_duration=4

    for i in tqdm(range(sample_no, sample_no+1)):

        time_step1_mfccs=[]
        time_step1_phns=[]
        y, sr = librosa.load(timit_train_wav[i], sr=None)
        phoneme =  open(timit_train_phns[i], "r").read()
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
                start_index = int(phoneme[k][0])
                end_index = int(phoneme[k][1])
                if(j>=start_index and j<=end_index):
                    phn_str = phoneme[k][2]
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


train1_mfccs, train1_phns = get_all_feature_phoneme(timit_train_wav, timit_train_phns)
# print(train1_mfccs.shape)
# print(train1_phns.shape)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(800,13)))
model.add(layers.Dropout(0.1))
model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(800,64)))
model.add(layers.Dropout(0.1))
model.add(layers.TimeDistributed(layers.Dense(64, activation="relu")))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(61, activation="softmax"))
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

BATCH_SIZE=32
EPOCHS=2

history = model.fit(np.array(train1_mfccs), np.array(train1_phns), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
test1_mfccs, test1_phns = get_all_feature_phoneme(timit_test_wav, timit_test_phns)

pred_cat = model.predict(np.array(test1_mfccs))
pred = np.argmax(pred_cat, axis=-1)
# print(pred)

y_te_true = np.argmax(np.array(test1_phns), -1)

# print(pred.shape)
# # print(y_te_true)
# print(np.array(y_te_true).shape)

height,width=pred.shape

acc_count=0
for i in range(height):
    for j in range(width):
        if(pred[i,j] == y_te_true[i,j]):
            acc_count = acc_count+1
accuracy=acc_count/(height*width)
print(f"Accuracy is on full test data is: {accuracy}")

# Take a random sample from test data
sample_no = 220
print(f'Test sample no from source: {timit_test_wav[sample_no]}')
y_test_timit, sr = librosa.load(timit_test_wav[i], sr=None)
# print(len(y_test_timit))
test_feature, test_phns=get_one_feature_phoneme(timit_test_wav, timit_test_phns, sample_no)

pred_cat = model.predict(np.array(test_feature))
pred_phn_for_net2 = pred_cat[0,:,:]
pred = np.argmax(pred_cat, axis=-1)
# print(pred)

y_te_true = np.argmax(np.array(test_phns), -1)

# print(pred.shape)
# # print(y_te_true)
# print(np.array(y_te_true).shape)

height,width=pred.shape

acc_count=0
for i in range(height):
    for j in range(width):
        if(pred[i,j] == y_te_true[i,j]):
#             if(pred[i,j]!=0):
                acc_count = acc_count+1
accuracy=acc_count/(height*width)
print(f"Accuracy for Test sample no from source: {arctic_test_wav[sample_no]} is: {accuracy}")

print("Predicted Phones are:")
print(pred)
print("True Phones are:")
print(y_te_true)


def get_all_mel_phoneme(arctic_train_wav2, arctic_train_phns2):
    from tqdm import tqdm
    train2_mel = []
    train2_phns = []

    for i in tqdm(range(len(arctic_train_wav2))):
        y, sr = librosa.load(arctic_train_wav2[i], sr=None)
        phoneme =  open(arctic_train_phns2[i], "r").read()
        phoneme = string_to_matrix_dict(phoneme)
        win = int(win_length*sr)
        hop = int(hop_length*sr)

        y_mel = librosa.feature.melspectrogram(y=y, sr=sr, win_length=win, hop_length=hop)

        count = 0
        for j in range(0,len(y),hop):
            count = count+1
            index = int(j/hop)
            train2_mel.append(y_mel[:,index])
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
                    train2_phns.append(phn_one_hot)
    #                 train2_phns.append(phn_label)
                    x=x+1
                    break
            if(x==0):
                phn_label = 0
                phn_one_hot = np.eye(len(phns))[phn_label]
                train2_phns.append(phn_one_hot) 
    #             train2_phns.append(phn_label)

    return train2_mel, train2_phns


def get_one_mel_phoneme(arctic_train_wav2, arctic_train_phns2, sample_no):
    from tqdm import tqdm
    train2_mel = []
    train2_phns = []

    for i in tqdm(range(sample_no, sample_no+1)):
        y, sr = librosa.load(arctic_train_wav2[i], sr=None)
        phoneme =  open(arctic_train_phns2[i], "r").read()
        phoneme = string_to_matrix_dict(phoneme)

        win = int(win_length*sr)
        hop = int(hop_length*sr)

        y_mel = librosa.feature.melspectrogram(y=y, sr=sr, win_length=win, hop_length=hop)

        count = 0
        for j in range(0,len(y),hop):
            count = count+1
            index = int(j/hop)
 
            train2_mel.append(y_mel[:,index])
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
                    train2_phns.append(phn_one_hot)
    #                 train2_phns.append(phn_label)
                    x=x+1
                    break
            if(x==0):
                phn_label = 0
                phn_one_hot = np.eye(len(phns))[phn_label]
                train2_phns.append(phn_one_hot) 
    #             train2_phns.append(phn_label)

    return train2_mel, train2_phns


train2_mel, train2_phns=get_all_mel_phoneme(arctic_train_wav2, arctic_train_phns2)
# print(len(train2_phns))
# print(len(train2_mel))
# print(np.array(train2_mel).shape)
# print(np.array(train2_phns).shape)

model = keras.Sequential()
model.add(layers.Dense(128, input_dim=len(phns), activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation='linear'))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()

BATCH_SIZE=32
EPOCHS=5

history=model.fit(np.array(train2_phns),np.array(train2_mel),batch_size=BATCH_SIZE,epochs=EPOCHS,validation_split=0.1,verbose=1)

sample_no=4
print(arctic_test_wav2[sample_no])
test2_mel, test2_phns=get_one_mel_phoneme(arctic_test_wav2, arctic_test_phns2, sample_no)

# Eval
pred_mel = model.predict(np.array(test2_phns))
pred_mel=pred_mel.T
print(np.array(test2_phns).shape)
print(pred_mel.shape)
# # print(np.array(test2_mel).shape)
# S_inv = librosa.feature.inverse.mel_to_stft(pred_mel, sr=sr)
# y_inv = librosa.griffinlim(S_inv)
# # ipd.Audio(y, rate=sr, autoplay=True) # load a local WAV file
sr=16000
y_inv=librosa.feature.inverse.mel_to_audio(pred_mel, sr=sr, win_length=400, hop_length=80)
print(len(y_inv))
print(len(y_inv)/sr)

import soundfile as sf
sf.write('output1.wav',y_inv, sr)

phns_mel = [[] for i in range(61)]
# phns_mel[0].append([1,2])
# phns_mel[0].append([2,2])
# phns_mel[1].append([2,2])
print(len(phns_mel))
print(phns_mel)
print(phns_mel[0])

from tqdm import tqdm
train2_mel = []
train2_phns = []

for i in tqdm(range(len(arctic_train_wav2))):
# for i in tqdm(range(1)):
    y, sr = librosa.load(arctic_train_wav2[i], sr=None)
    phoneme =  open(arctic_train_phns2[i], "r").read()
    phoneme = string_to_matrix_dict(phoneme)
#     phoneme=phoneme[2:len(phoneme)-1]
#     print(phoneme)
#     end_y = int(phoneme[len(phoneme)-1][1])
    win = int(win_length*sr)
    hop = int(hop_length*sr)
#     y=y[:end_y]
    y_mel = librosa.feature.melspectrogram(y=y, sr=sr, win_length=win, hop_length=hop)
#     print(y_mel)
#     print(y_mel.shape)
    count = 0
# #     print((phoneme[0][1]))
    for j in range(0,len(y),hop):
        count = count+1
        index = int(j/hop)
#         print(index)
#         train2_mel.append(y_mel[:,index])
        x=0
        for k in range(1,len(phoneme)-1):
            start_index = int(sr*(float(phoneme[k][0])))
            next_index = int(sr*(float(phoneme[k+1][0])))
#             print(start_index)
#             print(j, start_index)
            if(j>=start_index and j<=next_index):
                phn_str = phoneme[k+1][2]
                phn_label = phn2idx[phn_str]
                if(phn_label==44):
                    phn_label=0
                phn_one_hot = np.eye(len(phns))[phn_label]
                phns_mel[phn_label].append(y_mel[:,index])
#                 train2_phns.append(phn_one_hot)
#                 train2_phns.append(phn_label)
                x=x+1
                break
        if(x==0):
            phn_label = 0
            phn_one_hot = np.eye(len(phns))[phn_label]
            phns_mel[phn_label].append(y_mel[:,index])
#             train2_phns.append(phn_one_hot) 
#             train2_phns.append(phn_label)

# print(count)
# print(len(train2_mel))
# print(train2_phns)        

#print(train2_mf2l)
# print(len(train2_phns))      

# print(len(y))    
# print(phoneme)

# length=0
# for i in range(len(phns_mel)):
#     length = length+ len(phns_mel[i])
# print(length)
# print(len(phns_mel[4]))

print(len(phns_mel))
print(len(phns_mel[0]))
print(np.array(phns_mel[0]).shape)
phns_final_mel = np.zeros((61,128))

for i in range(len(phns_mel)):
#     print(len(phns_mel[i]))
    if(len(phns_mel[i]) != 0):
        phns_final_mel[i]=sum(phns_mel[i])/len(phns_mel[i])

for i in tqdm(range(len(phns_mel))):
#     print(len(phns_mel[i]))
    if(len(phns_mel[i]) != 0):
        h,w = np.array(phns_mel[i]).shape
        norm_vector = np.zeros((h,1))
        for j in range(h):
            norm_vector[j] = np.linalg.norm(np.array(phns_mel[i])[j,:])
        index=np.argmax(norm_vector)
        phns_final_mel[i]=np.array(phns_mel[i])[index,:]

print(phns_final_mel.shape)
print(phns_final_mel[0].shape)


def get_output_mel(arctic_test_wav2, arctic_test_phns2, sample_no):
    from tqdm import tqdm
    test2_mel = []
    test2_phns = []
    output_mel=[]

    # for i in tqdm(range(len(arctic_test_wav2))):
    for i in tqdm(range(sample_no, sample_no+1)):
        y, sr = librosa.load(arctic_test_wav2[i], sr=None)
        phoneme =  open(arctic_test_phns2[i], "r").read()
        phoneme = string_to_matrix_dict(phoneme)
    #     phoneme=phoneme[2:len(phoneme)-1]
    #     print(phoneme)
    #     end_y = int(phoneme[len(phoneme)-1][1])
        win = int(win_length*sr)
        hop = int(hop_length*sr)
    #     y=y[:end_y]
        y_mel = librosa.feature.melspectrogram(y=y, sr=sr, win_length=win, hop_length=hop)
    #     print(y_mel)
    #     print(y_mel.shape)
        count = 0
    # #     print((phoneme[0][1]))
        for j in range(0,len(y),hop):
            count = count+1
            index = int(j/hop)
    #         print(index)
            test2_mel.append(y_mel[:,index])
            x=0
            for k in range(1,len(phoneme)-1):
                start_index = int(sr*(float(phoneme[k][0])))
                next_index = int(sr*(float(phoneme[k+1][0])))
    #             print(start_index)
    #             print(j, start_index)
                if(j>=start_index and j<=next_index):
                    phn_str = phoneme[k+1][2]
                    phn_label = phn2idx[phn_str]
                    if(phn_label==44):
                        phn_label=0
                    phn_one_hot = np.eye(len(phns))[phn_label]
                    output_mel.append(phns_final_mel[phn_label])
                    test2_phns.append(phn_one_hot)
    #                 test2_phns.append(phn_label)
                    x=x+1
                    break
            if(x==0):
                phn_label = 0
                phn_one_hot = np.eye(len(phns))[phn_label]
                output_mel.append(phns_final_mel[phn_label])
                test2_phns.append(phn_one_hot) 
    #             test2_phns.append(phn_label)
    # print(count)
    # print(len(test2_mel))
    # print(train2_phns)        

    #print(train2_mf2l)
    # print(len(test2_phns))      

    # print(len(y))    
    # print(phoneme)
    return output_mel

sample_no=4
print(arctic_test_wav2[sample_no])
output_mel=get_output_mel(arctic_test_wav2, arctic_test_phns2, sample_no)

output_mel=np.array(output_mel).T
print((output_mel).shape)
# print((output_mel))

y_inv=librosa.feature.inverse.mel_to_audio(output_mel, sr=sr, win_length=400, hop_length=80)
print(len(y_inv))
print(len(y_inv)/sr)

sf.write('output2.wav',y_inv, sr)




