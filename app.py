
import joblib
import tensorflow as tf
model = tf.keras.models.load_model("./models/model.h5")
model2 = tf.keras.models.load_model("./models/model_convolution.h5")
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import pandas as pd
import IPython.display
from ipywidgets import interact, interactive, fixed 
from PIL import Image
import tensorflow as tf
from scipy.io import wavfile
import scipy.signal as sps
import glob
import copy
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.ndimage
from tqdm import tqdm
from pydub import AudioSegment
from pydub.utils import make_chunks
import time
import io
import math
import re
import soundfile as sf
import librosa
from audio_recorder_streamlit import audio_recorder
wi=200

def main():
    st.sidebar.markdown("## Noise Cancellation")
    page = st.sidebar.selectbox("Pages", ["About","Our Team","Noise cancellation","Other attempts"])
    #about page
    if page=="About":
        st.title("IITISoc PROJECT")
        st.text("DOMAIN :MACHINE LEARNING ")
        st.subheader("PROJECT TASK: To remove noise from a given audio sample with the help of deep learning algorithms")
        st.header("Our Solution \n")
        st.subheader("we used convolution layers in our model to extract features from the audio sample,we convert the audio into an image form known as a spectogram which contains information about the frequency and amplitute of the constituents sounds in the audio.These images were trained in the model to predict the cleaned sample.")
        st.subheader("To test the model ,please go to Noise cancellation page")
        st.subheader("You can also check \"Other attempts \"")

    # team page
    if page =="Our Team":    
        st.header("Our team")
        col1, col2 = st.columns([1,1],gap="large")
        
        with col1:
            st.text("Name : Abhinav gangil")
            st.text("Email: cse220001002@iiti.ac.in")
            st.text("roll no.: 220001002")
            st.image(Image.open("./images/abhinav.jpeg"),width=wi)
        with col2:
            st.text("Name : Aadish Jain")
            st.text("Email: cse220001001@iiti.ac.in")
            st.text("roll no.: 220001001")
            st.image(Image.open("./images/aadish.jpeg"),width=wi)   
        col1, col2 = st.columns([1,1],gap="large")
        
        with col1:
            st.text("Name : Aditi wekhande")
            st.text("Email: cse220001003@iiti.ac.in")
            st.text("roll no.: 220001003")
            st.image(Image.open("./images/aditi.jpeg"),width=wi)
        with col2:
            st.text("Name : Aman Poddar")
            st.text("Email: ee220002006@iiti.ac.in")
            st.text("roll no.: 220002006")
            st.image(Image.open("./images/aman.png"),width=wi)      
  

    # other model
    if page== "Other attempts":
        st.title("Noise Cancellation(Previous approch)")
        audio_files=audio_recorder(sample_rate=16_000,icon_size="2x")
        st.write("or")
        audio_file = st.file_uploader("Upload audio file", type=["wav"])
        if(audio_files !=None):
            noise,_=librosa.load(io.BytesIO(audio_files))
        elif(audio_file!=None):
            noise,_=librosa.load(audio_file)
        if st.button('Process', use_container_width=True):
            if audio_file is not None:
                Frame_size=510
                Hop_length=256
                mini_batch=8000
                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                def sound_to_number_converter(list):
                    converted_sound=[]
                    for i in list:
                        val,_=librosa.load(i)
                        converted_sound.append(np.array(val))
                    return converted_sound

                    #perform stft transform on a list containing numbered_data_of_audio input-numbered_data list output-list_of_stft_transformed_of_respective_Audio
                def stft_transformer(list):
                    list_stft=[]
                    for i in list:
                        val=librosa.stft(i,n_fft=Frame_size,hop_length=Hop_length)
                        list_stft.append(val)
                    return list_stft
                    #return magnitude and real part of stft transformed data input-list of stft_numbered_data output-first list of magnitude and second list of real part
                def get_phase(list):
                    magnitude=[]
                    first_val=[]

                    for i in list:
                        val2=tf.abs(i).numpy()
                        val3=tf.math.real(i).numpy()
                        first_val.append(val3)
                        magnitude.append(val2)
                    return magnitude,first_val
                    #plot waveform input-path of signal ,title  output-image of waveform
                def plot_waveForm(signal,title):
                    
                    signal_array,signal_fr=librosa.load(signal)
                    signal_array=np.array(signal_array)
                    plt.figure(figsize=(15,17))
                    plt.subplot(3,1,1)
                    librosa.display.waveshow(signal_array,alpha=0.5)
                    plt.title(title)
                    plt.ylim((-1,1))
                    return signal_array
                    # pairing
                def pair(clean,noise):
                    return noise,clean
                    # mini_batch
                    #creates small patches of the signal of size mini_batch input-signal(normal array type) output-batched_list of the signal ,difference
                def mini_batch(signal,mini_batch=mini_batch):
                    signalList=[]
                    i=0
                    for i in range(0,int(len(signal))-mini_batch,mini_batch):
                        li=signal[i:i+mini_batch]
                        signalList.append(li)
                    signalList.append(signal[-mini_batch:])
                    diff=len(signal)-i-mini_batch
                    return np.array(tf.stack(signalList)),diff
                    #batches a list of numerical_data into batches using mini_batch function and returns list of batched data with there corresponding difference.
                def batching_numerical_data(signal_data):
                    list=[]
                    list_diff=[]
                    for i in signal_data:
                        li,_=mini_batch(i)
                        list.append(li)
                        list_diff.append(_)
                    return list,list_diff
                    #perform stft transform on a list of data
                def batching_stft_transformation(signal_data):
                    list=[]
                    for i in signal_data:
                        list.append(stft_transformer(i))
                    return list
                    #convert list of numbered_data to tensors
                def converting_to_tensor(signal):
                    list=[]
                    for i in signal:
                        list.append(tf.constant(i))
                    list=tf.stack(list)
                    return list
                def matching_correct(clean_speech_list,noise_list,decibal):
                    noise=[]
                    for i in clean_speech_list:
                        num=int(re.findall("\d+",i)[2])
                        for j in noise_list:
                            val=int(re.findall("\d+",j)[2])
                            deci=int(re.findall("\d+",j)[3])
                            if(val==num and deci==decibal  ):
                                noise.append(j)
                                break
                                
                    return noise   
                def making_common_list(data):
                    c_list=[]
                    for i in data:
                        for j in i:
                            c_list.append(j)
                    return c_list
                batchsize=50
                def testing(model,noise,batching_size,Frame_size,Hoplength,_):
                        
                        
                        noise,noise_diff=mini_batch(noise,batching_size)
                        list_stft=[]
                        for i in noise:
                            val=librosa.stft(i,n_fft=Frame_size,hop_length=Hoplength)
                            list_stft.append(val)
                        list_stftmag,list_stftphase=get_phase(list_stft)
                        list=[]
                        shape=(list_stftmag[0].shape[0],list_stftmag[0].shape[1],1)
                        for j in list_stftmag:
                            j=j.reshape(shape)
                            list.append(j) 
                        predict_val=model.predict((tf.constant(list)))   
                        new_predict_val=[]
                        for i in range(len(predict_val)):
                            nl=[]
                            for j in range(len(predict_val[i])):
                                freq_list=[complex(predict_val[i][j][y]*np.sin(list_stftphase[i][j][y]),predict_val[i][j][y]*np.cos(list_stftphase[i][j][y])) for y in range(len(predict_val[i][j]))]
                                nl.append(np.array(freq_list) )  
                            new_predict_val.append(np.array(nl)) 
                        list_audio=[]
                        for i in new_predict_val:
                            istft_val=librosa.istft(i,n_fft=Frame_size,hop_length=Hoplength)
                            list_audio.append(istft_val)
                        li=np.array([])
                        for i in range(len(list_audio)):
                            li=np.concatenate((li,list_audio[i]))
                        li=np.concatenate((li,list_audio[-1][-noise_diff:])) 
                        
                        sf.write(f"prediction_model2.wav", li, _)
                        
                        return li
                
                testing(model2,noise,8000,Frame_size,Hop_length,_)
                for percent_complete in range(100):
                    time.sleep(0.02)
                    my_bar.progress(percent_complete + 1, text=progress_text)

                my_bar.empty()
                st.write('Given Audio File')
                st.audio(audio_file, format='wav')

                # Read the audio file
                audio_file = open('prediction_model2.wav', 'rb')
                audio_bytes = audio_file.read()

                # Display the audio player
                st.write('Processed Audio File')
                st.audio(audio_bytes, format='wav')     
            elif(audio_files is not None):
                Frame_size=510
                Hop_length=256
                mini_batch=8000
                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                def sound_to_number_converter(list):
                    converted_sound=[]
                    for i in list:
                        val,_=librosa.load(i)
                        converted_sound.append(np.array(val))
                    return converted_sound

                    #perform stft transform on a list containing numbered_data_of_audio input-numbered_data list output-list_of_stft_transformed_of_respective_Audio
                def stft_transformer(list):
                    list_stft=[]
                    for i in list:
                        val=librosa.stft(i,n_fft=Frame_size,hop_length=Hop_length)
                        list_stft.append(val)
                    return list_stft
                    #return magnitude and real part of stft transformed data input-list of stft_numbered_data output-first list of magnitude and second list of real part
                def get_phase(list):
                    magnitude=[]
                    first_val=[]

                    for i in list:
                        val2=tf.abs(i).numpy()
                        val3=tf.math.real(i).numpy()
                        first_val.append(val3)
                        magnitude.append(val2)
                    return magnitude,first_val
                    #plot waveform input-path of signal ,title  output-image of waveform
                def plot_waveForm(signal,title):
                    
                    signal_array,signal_fr=librosa.load(signal)
                    signal_array=np.array(signal_array)
                    plt.figure(figsize=(15,17))
                    plt.subplot(3,1,1)
                    librosa.display.waveshow(signal_array,alpha=0.5)
                    plt.title(title)
                    plt.ylim((-1,1))
                    return signal_array
                    # pairing
                def pair(clean,noise):
                    return noise,clean
                    # mini_batch
                    #creates small patches of the signal of size mini_batch input-signal(normal array type) output-batched_list of the signal ,difference
                def mini_batch(signal,mini_batch=mini_batch):
                    signalList=[]
                    i=0
                    for i in range(0,int(len(signal))-mini_batch,mini_batch):
                        li=signal[i:i+mini_batch]
                        signalList.append(li)
                    signalList.append(signal[-mini_batch:])
                    diff=len(signal)-i-mini_batch
                    return np.array(tf.stack(signalList)),diff
                    #batches a list of numerical_data into batches using mini_batch function and returns list of batched data with there corresponding difference.
                def batching_numerical_data(signal_data):
                    list=[]
                    list_diff=[]
                    for i in signal_data:
                        li,_=mini_batch(i)
                        list.append(li)
                        list_diff.append(_)
                    return list,list_diff
                    #perform stft transform on a list of data
                def batching_stft_transformation(signal_data):
                    list=[]
                    for i in signal_data:
                        list.append(stft_transformer(i))
                    return list
                    #convert list of numbered_data to tensors
                def converting_to_tensor(signal):
                    list=[]
                    for i in signal:
                        list.append(tf.constant(i))
                    list=tf.stack(list)
                    return list
                def matching_correct(clean_speech_list,noise_list,decibal):
                    noise=[]
                    for i in clean_speech_list:
                        num=int(re.findall("\d+",i)[2])
                        for j in noise_list:
                            val=int(re.findall("\d+",j)[2])
                            deci=int(re.findall("\d+",j)[3])
                            if(val==num and deci==decibal  ):
                                noise.append(j)
                                break
                                
                    return noise   
                def making_common_list(data):
                    c_list=[]
                    for i in data:
                        for j in i:
                            c_list.append(j)
                    return c_list
                batchsize=50
                def testing(model,noise,batching_size,Frame_size,Hoplength,_):
                        
                        
                        noise,noise_diff=mini_batch(noise,batching_size)
                        list_stft=[]
                        for i in noise:
                            val=librosa.stft(i,n_fft=Frame_size,hop_length=Hoplength)
                            list_stft.append(val)
                        list_stftmag,list_stftphase=get_phase(list_stft)
                        list=[]
                        shape=(list_stftmag[0].shape[0],list_stftmag[0].shape[1],1)
                        for j in list_stftmag:
                            j=j.reshape(shape)
                            list.append(j) 
                        predict_val=model.predict((tf.constant(list)))   
                        new_predict_val=[]
                        for i in range(len(predict_val)):
                            nl=[]
                            for j in range(len(predict_val[i])):
                                freq_list=[complex(predict_val[i][j][y]*np.sin(list_stftphase[i][j][y]),predict_val[i][j][y]*np.cos(list_stftphase[i][j][y])) for y in range(len(predict_val[i][j]))]
                                nl.append(np.array(freq_list) )  
                            new_predict_val.append(np.array(nl)) 
                        list_audio=[]
                        for i in new_predict_val:
                            istft_val=librosa.istft(i,n_fft=Frame_size,hop_length=Hoplength)
                            list_audio.append(istft_val)
                        li=np.array([])
                        for i in range(len(list_audio)):
                            li=np.concatenate((li,list_audio[i]))
                        li=np.concatenate((li,list_audio[-1][-noise_diff:])) 
                        
                        sf.write(f"prediction_model2.wav", li, _)
                        
                        return li
                
                testing(model2,noise,8000,Frame_size,Hop_length,_)
                for percent_complete in range(100):
                    time.sleep(0.02)
                    my_bar.progress(percent_complete + 1, text=progress_text)

                my_bar.empty()
                st.write('Given Audio File')
                st.audio(audio_files, format='wav')

                # Read the audio file
                audio_file = open('prediction_model2.wav', 'rb')
                audio_bytes = audio_file.read()

                # Display the audio player
                st.write('Processed Audio File')
                st.audio(audio_bytes, format='wav')
            

    if page == "Noise cancellation":
        st.title("Noise Cancellation")
        audio_files=audio_recorder(sample_rate=16_000,icon_size="2x")
        st.write("or")
        audio_file = st.file_uploader("Upload audio file", type=["wav"])
        if st.button('Process', use_container_width=True):
            if audio_files is not None:
                audio_bytes = audio_files # Read the uploaded file
                st.write('Given Audio File')
                st.audio(audio_bytes)
                sampling_rate, data = wavfile.read(io.BytesIO(audio_bytes))  # Read the audio data using wavfile.read
                rate = 16000
                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                FFT_LENGTH = 512
                WINDOW_LENGTH = 512
                WINDOW_STEP = int(WINDOW_LENGTH / 2)
                phaseMax = 3.141592653589793 
                phaseMin = -3.141592653589793
                magnitudeMax = 2211683.973249525
                magnitudeMin = 0.0
                def amplifyMagnitudeByLog(d):
                    return 188.301 * math.log10(d + 1)

                def weakenAmplifiedMagnitude(d):
                    return math.pow(10, d/188.301)-1

                def generateLinearScale(magnitudePixels, phasePixels, 
                                        magnitudeMin, magnitudeMax, phaseMin, phaseMax):
                    height = magnitudePixels.shape[0]
                    width = magnitudePixels.shape[1]
                    magnitudeRange = magnitudeMax - magnitudeMin
                    phaseRange = phaseMax - phaseMin
                    rgbArray = np.zeros((height, width, 3), 'uint8')
                
                    for w in range(width):
                        for h in range(height):
                            magnitudePixels[h,w] = (magnitudePixels[h,w] - magnitudeMin) / (magnitudeRange) * 255 * 2
                            magnitudePixels[h,w] = amplifyMagnitudeByLog(magnitudePixels[h,w])
                            phasePixels[h,w] = (phasePixels[h,w] - phaseMin) / (phaseRange) * 255
                            red = 255 if magnitudePixels[h,w] > 255 else magnitudePixels[h,w]
                            green = (magnitudePixels[h,w] - 255) if magnitudePixels[h,w] > 255 else 0
                            blue = phasePixels[h,w]
                            rgbArray[h,w,0] = int(red)
                            rgbArray[h,w,1] = int(green)
                            rgbArray[h,w,2] = int(blue)
                    return rgbArray

                def recoverLinearScale(rgbArray, magnitudeMin, magnitudeMax, 
                                    phaseMin, phaseMax):
                    width = rgbArray.shape[1]
                    height = rgbArray.shape[0]
                    magnitudeVals = rgbArray[:,:,0].astype(float) + rgbArray[:,:,1].astype(float)
                    phaseVals = rgbArray[:,:,2].astype(float)
                    phaseRange = phaseMax - phaseMin
                    magnitudeRange = magnitudeMax - magnitudeMin
                    for w in range(width):
                        for h in range(height):
                            phaseVals[h,w] = (phaseVals[h,w] / 255 * phaseRange) + phaseMin
                            magnitudeVals[h,w] = weakenAmplifiedMagnitude(magnitudeVals[h,w])
                            magnitudeVals[h,w] = (magnitudeVals[h,w] / (255*2) * magnitudeRange) + magnitudeMin
                    return magnitudeVals, phaseVals

                def generateSpectrogramForWave(signal):
                    start_time = time.time()
                    magnitudeMin = 0.0
                    magnitudeMax = 2211683.973249525
                    phaseMin = -3.141592653589793
                    phaseMax = 3.141592653589793
                    buffer = np.zeros(int(signal.size + WINDOW_STEP - (signal.size % WINDOW_STEP)))
                    buffer[0:len(signal)] = signal
                    height = int(FFT_LENGTH / 2 + 1)
                    width = int(len(buffer) / (WINDOW_STEP) - 1)
                    magnitudePixels = np.zeros((height, width))
                    phasePixels = np.zeros((height, width))

                    for w in range(width):
                        buff = np.zeros(FFT_LENGTH)
                        stepBuff = buffer[w*WINDOW_STEP:w*WINDOW_STEP + WINDOW_LENGTH]
                        # apply hanning window
                        stepBuff = stepBuff * np.hanning(WINDOW_LENGTH)
                        buff[0:len(stepBuff)] = stepBuff
                        #buff now contains windowed signal with step length and padded with zeroes to the end
                        fft = np.fft.rfft(buff)
                        for h in range(len(fft)):
                            magnitude = math.sqrt(fft[h].real**2 + fft[h].imag**2)
                            if magnitude > magnitudeMax:
                                magnitudeMax = magnitude 
                            if magnitude < magnitudeMin:
                                magnitudeMin = magnitude 

                            phase = math.atan2(fft[h].imag, fft[h].real)
                            if phase > phaseMax:
                                phaseMax = phase
                            if phase < phaseMin:
                                phaseMin = phase
                            magnitudePixels[height-h-1,w] = magnitude
                            phasePixels[height-h-1,w] = phase
                    rgbArray = generateLinearScale(magnitudePixels, phasePixels,
                                                magnitudeMin, magnitudeMax, phaseMin, phaseMax)
                    
                    
                    elapsed_time = time.time() - start_time
                    print('%.2f' % elapsed_time, 's', sep='')
                    img = Image.fromarray(rgbArray, 'RGB')
                    return img

                def recoverSignalFromSpectrogram(numpyarray):
                    data = np.array( numpyarray, dtype='uint8' )
                    width = data.shape[1]
                    height = data.shape[0]
                    magnitudeVals, phaseVals \
                    = recoverLinearScale(data, magnitudeMin, magnitudeMax, phaseMin, phaseMax)
                    recovered = np.zeros(WINDOW_LENGTH * width // 2 + WINDOW_STEP, dtype=np.int16)
                    recovered = np.array(recovered,dtype=np.int16)
                    
                    for w in range(width):
                        toInverse = np.zeros(height, dtype=np.complex_)
                        for h in range(height):
                            magnitude = magnitudeVals[height-h-1,w]
                            phase = phaseVals[height-h-1,w]
                            toInverse[h] = magnitude * math.cos(phase) + (1j * magnitude * math.sin(phase))
                        signal = np.fft.irfft(toInverse)
                        recovered[w*WINDOW_STEP:w*WINDOW_STEP + WINDOW_LENGTH] += signal[:WINDOW_LENGTH].astype(np.int16)
                    return recovered
                number_of_samples = round(len(data) * float(rate) / sampling_rate)
                data = sps.resample(data, number_of_samples)
                data = np.asarray(data, dtype=np.int16)
                filename = "processed_audio.wav"
                wavfile.write(filename,rate,data)
                myaudio = AudioSegment.from_file(filename)
                chunk_length_ms = 1000 # pydub calculates in millisec
                chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
                    
                for i, chunk in enumerate(chunks):
                        
                    chunk_name = "chunk{0}.wav".format(i)
                    name = chunk_name
                    
                    chunk.export(name, format="wav")
                    
                    rate, data = wavfile.read(name)
                    
                    
                    if len(data.shape) >= 2 and data.size > 0:
                        if data.shape[-1] > 1:
                            data = data.mean(axis=-1)
                        else:
                            data = np.reshape(data, data.shape[:-1])
                    img = generateSpectrogramForWave(data)
                    np.save(chunk_name[:-4]+'.npy', img) # save
                model.summary()
                ROW = 257
                COL = 62
                test_pred = []
                count = 0
                rate = 16000
                for i, _ in enumerate(chunks):
                    filename = 'chunk{0}'.format(i)+'.npy'
                    img_test = np.load(filename)
                    row_,col_,_ = img_test.shape
                    
                    if col_ < COL:
                        continue
                    
                    print(img_test.shape)
                    
                    img_test = img_test/255
                    img_test = img_test.reshape(-1, ROW,COL,3)
                    decoded_imgs = model.predict(img_test) #predict
                    decoded_imgs = decoded_imgs.reshape(ROW,COL,3)
                    decoded_imgs = decoded_imgs*255
                    decoded_imgs = decoded_imgs.astype(np.int16)
                    data = recoverSignalFromSpectrogram(decoded_imgs) # save predicted audio
                    file = "testpred_{}".format(count)+'.wav' #created file
                    scipy.io.wavfile.write(file, rate, data) #wrote the file with 'data'
                    test_pred.append(file) #saves file in array
                    count = count+1
                sound = 0
                for i in range(len(test_pred)):
                    print(test_pred[i])
                    sound += AudioSegment.from_wav(test_pred[i])
                sound.export("soundJoined1.wav", format="wav")
                for percent_complete in range(100):
                    time.sleep(0.02)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                my_bar.empty()
                # Read the audio file
                audio_file = open('soundJoined1.wav', 'rb')
                audio_bytes = audio_file.read()

                # Display the audio player
                st.write('Processed Audio File')
                st.audio(audio_bytes, format='wav')
            
            
            elif audio_file is not None:
                audio_bytes = audio_file.read()  # Read the uploaded file
                st.write('Given Audio File')
                st.audio(audio_bytes)
                sampling_rate, data = wavfile.read(io.BytesIO(audio_bytes))  # Read the audio data using wavfile.read
                rate = 16000
                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                FFT_LENGTH = 512
                WINDOW_LENGTH = 512
                WINDOW_STEP = int(WINDOW_LENGTH / 2)
                phaseMax = 3.141592653589793 
                phaseMin = -3.141592653589793
                magnitudeMax = 2211683.973249525
                magnitudeMin = 0.0
                def amplifyMagnitudeByLog(d):
                    return 188.301 * math.log10(d + 1)

                def weakenAmplifiedMagnitude(d):
                    return math.pow(10, d/188.301)-1

                def generateLinearScale(magnitudePixels, phasePixels, 
                                        magnitudeMin, magnitudeMax, phaseMin, phaseMax):
                    height = magnitudePixels.shape[0]
                    width = magnitudePixels.shape[1]
                    magnitudeRange = magnitudeMax - magnitudeMin
                    phaseRange = phaseMax - phaseMin
                    rgbArray = np.zeros((height, width, 3), 'uint8')
                
                    for w in range(width):
                        for h in range(height):
                            magnitudePixels[h,w] = (magnitudePixels[h,w] - magnitudeMin) / (magnitudeRange) * 255 * 2
                            magnitudePixels[h,w] = amplifyMagnitudeByLog(magnitudePixels[h,w])
                            phasePixels[h,w] = (phasePixels[h,w] - phaseMin) / (phaseRange) * 255
                            red = 255 if magnitudePixels[h,w] > 255 else magnitudePixels[h,w]
                            green = (magnitudePixels[h,w] - 255) if magnitudePixels[h,w] > 255 else 0
                            blue = phasePixels[h,w]
                            rgbArray[h,w,0] = int(red)
                            rgbArray[h,w,1] = int(green)
                            rgbArray[h,w,2] = int(blue)
                    return rgbArray

                def recoverLinearScale(rgbArray, magnitudeMin, magnitudeMax, 
                                    phaseMin, phaseMax):
                    width = rgbArray.shape[1]
                    height = rgbArray.shape[0]
                #     print(phaseMax,phaseMin)
                    magnitudeVals = rgbArray[:,:,0].astype(float) + rgbArray[:,:,1].astype(float)
                    phaseVals = rgbArray[:,:,2].astype(float)
                    phaseRange = phaseMax - phaseMin
                    magnitudeRange = magnitudeMax - magnitudeMin
                    for w in range(width):
                        for h in range(height):
                            phaseVals[h,w] = (phaseVals[h,w] / 255 * phaseRange) + phaseMin
                            magnitudeVals[h,w] = weakenAmplifiedMagnitude(magnitudeVals[h,w])
                            magnitudeVals[h,w] = (magnitudeVals[h,w] / (255*2) * magnitudeRange) + magnitudeMin
                    return magnitudeVals, phaseVals

                def generateSpectrogramForWave(signal):
                    start_time = time.time()
                    magnitudeMin = 0.0
                    magnitudeMax = 2211683.973249525
                    phaseMin = -3.141592653589793
                    phaseMax = 3.141592653589793
                    buffer = np.zeros(int(signal.size + WINDOW_STEP - (signal.size % WINDOW_STEP)))
                    buffer[0:len(signal)] = signal
                    height = int(FFT_LENGTH / 2 + 1)
                    width = int(len(buffer) / (WINDOW_STEP) - 1)
                    magnitudePixels = np.zeros((height, width))
                    phasePixels = np.zeros((height, width))

                    for w in range(width):
                        buff = np.zeros(FFT_LENGTH)
                        stepBuff = buffer[w*WINDOW_STEP:w*WINDOW_STEP + WINDOW_LENGTH]
                        # apply hanning window
                        stepBuff = stepBuff * np.hanning(WINDOW_LENGTH)
                        buff[0:len(stepBuff)] = stepBuff
                        #buff now contains windowed signal with step length and padded with zeroes to the end
                        fft = np.fft.rfft(buff)
                        for h in range(len(fft)):
                            magnitude = math.sqrt(fft[h].real**2 + fft[h].imag**2)
                            if magnitude > magnitudeMax:
                                magnitudeMax = magnitude 
                            if magnitude < magnitudeMin:
                                magnitudeMin = magnitude 

                            phase = math.atan2(fft[h].imag, fft[h].real)
                            if phase > phaseMax:
                                phaseMax = phase
                            if phase < phaseMin:
                                phaseMin = phase
                            magnitudePixels[height-h-1,w] = magnitude
                            phasePixels[height-h-1,w] = phase
                    rgbArray = generateLinearScale(magnitudePixels, phasePixels,
                                                magnitudeMin, magnitudeMax, phaseMin, phaseMax)
                    
                    
                    elapsed_time = time.time() - start_time
                    print('%.2f' % elapsed_time, 's', sep='')
                    img = Image.fromarray(rgbArray, 'RGB')
                    return img

                def recoverSignalFromSpectrogram(numpyarray):
                    data = np.array( numpyarray, dtype='uint8' )
                    width = data.shape[1]
                    height = data.shape[0]
                    magnitudeVals, phaseVals \
                    = recoverLinearScale(data, magnitudeMin, magnitudeMax, phaseMin, phaseMax)
                    recovered = np.zeros(WINDOW_LENGTH * width // 2 + WINDOW_STEP, dtype=np.int16)
                    recovered = np.array(recovered,dtype=np.int16)
                    
                    for w in range(width):
                        toInverse = np.zeros(height, dtype=np.complex_)
                        for h in range(height):
                            magnitude = magnitudeVals[height-h-1,w]
                            phase = phaseVals[height-h-1,w]
                            toInverse[h] = magnitude * math.cos(phase) + (1j * magnitude * math.sin(phase))
                        signal = np.fft.irfft(toInverse)
                        recovered[w*WINDOW_STEP:w*WINDOW_STEP + WINDOW_LENGTH] += signal[:WINDOW_LENGTH].astype(np.int16)
                    return recovered
                number_of_samples = round(len(data) * float(rate) / sampling_rate)
                data = sps.resample(data, number_of_samples)
                data = np.asarray(data, dtype=np.int16)
                filename = "processed_audio.wav"
                wavfile.write(filename,rate,data)
                myaudio = AudioSegment.from_file(filename)
                chunk_length_ms = 1000 # pydub calculates in millisec
                chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
                    
                for i, chunk in enumerate(chunks):
                        
                    chunk_name = "chunk{0}.wav".format(i)
                    name = chunk_name
                    
                    chunk.export(name, format="wav")
                    
                    rate, data = wavfile.read(name)
                    
                    
                    if len(data.shape) >= 2 and data.size > 0:
                        if data.shape[-1] > 1:
                            data = data.mean(axis=-1)
                        else:
                            data = np.reshape(data, data.shape[:-1])
                    img = generateSpectrogramForWave(data)
                    np.save(chunk_name[:-4]+'.npy', img) # save
                model.summary()
                ROW = 257
                COL = 62

                test_pred = []

                
                count = 0
                rate = 16000
                for i, _ in enumerate(chunks):
                    filename = 'chunk{0}'.format(i)+'.npy'
                    img_test = np.load(filename)
                    row_,col_,_ = img_test.shape
                    
                    if col_ < COL:
                        continue
                    
                    print(img_test.shape)
                    
                    img_test = img_test/255
                    img_test = img_test.reshape(-1, ROW,COL,3)
                    decoded_imgs = model.predict(img_test) #predict
                    decoded_imgs = decoded_imgs.reshape(ROW,COL,3)
                    decoded_imgs = decoded_imgs*255
                    decoded_imgs = decoded_imgs.astype(np.int16)
                    data = recoverSignalFromSpectrogram(decoded_imgs) # save predicted audio
                    file = "testpred_{}".format(count)+'.wav' #created file
                    scipy.io.wavfile.write(file, rate, data) #wrote the file with 'data'
                    test_pred.append(file) #saves file in array
                    count = count+1

                sound = 0
                    
                for i in range(len(test_pred)):
                    print(test_pred[i])
                    sound += AudioSegment.from_wav(test_pred[i])
                sound.export("soundJoined1.wav", format="wav")

                for percent_complete in range(100):
                    time.sleep(0.02)
                    my_bar.progress(percent_complete + 1, text=progress_text)

                my_bar.empty()

                # Read the audio file
                audio_file = open('soundJoined1.wav', 'rb')
                audio_bytes = audio_file.read()

                # Display the audio player
                st.write('Processed Audio File')
                st.audio(audio_bytes, format='wav')
            



# def identify():
if __name__ == "__main__":
    main()


