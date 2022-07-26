# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:44:15 2019

@author: sayadav
"""
import os
import numpy as np
from keras.models import load_model
from keras import backend as K
import pickle

def Pickle_LoadObject(filename):
    return pickle.load(open(filename, 'rb'))

def decode_sequence(input_seq, encoder_model, 
                    decoder_model, 
                    target_token_index, 
                    num_decoder_tokens,
                    reverse_target_char_index,
                    max_decoder_seq_length):
    #Encode the input as state vector
    state_value = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    
    # Populate the first character of target sequence with the start character
    target_seq[0, 0, target_token_index['\t']] = 1.
    
    # Sampling loop for batch sequence
    # to simplify, here we assume a batch size of 1
    stop_condition = False
    decoded_santence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + state_value)
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_santence += sampled_char
        
        # Exit condition: either hit max length
        # or find the stop character
        if (sampled_char == '\n' or len(decoded_santence) > max_decoder_seq_length):
            stop_condition = True
        
        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        
        #update the states
        state_value = [h, c]
        
    return decoded_santence

def convert_english_to_hindi(APP_ROOT, text):
    import urllib.parse
    text = urllib.parse.unquote_plus(text)
    # prepare encoder model
    workingFolder = os.path.join(APP_ROOT,'models','nlp','Sequence to Sequence')
    
    with open(os.path.join(workingFolder, 'eng-hin_model_data_dict.pkl'),'rb') as f:
        model_data_dict = pickle.load(f)

    encoder_model = model_data_dict['encoder_model']
    max_encoder_seq_length = model_data_dict['max_encoder_seq_length']
    num_encoder_tokens = model_data_dict['num_encoder_tokens']
    input_token_index = model_data_dict['input_token_index']
    max_encoder_seq_length = model_data_dict['max_encoder_seq_length']
    
    decoder_model = model_data_dict['decoder_model']
    num_decoder_tokens = model_data_dict['num_decoder_tokens']
    target_token_index = model_data_dict['target_token_index']
    reverse_target_char_index = model_data_dict['reverse_target_char_index']
    max_decoder_seq_length = model_data_dict['max_decoder_seq_length']

    input_texts = []
    input_texts.append(text)
    
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    
    # preparing the training data
    for i, input_text in enumerate(input_texts):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        
        # set the remaining character as space so set the index of ' '
        encoder_input_data[i,t+1:,input_token_index[' ']] = 1.
    
    input_seq = encoder_input_data[0:1]
    decoded_sentence = decode_sequence(input_seq,encoder_model,
                                       decoder_model,
                                       target_token_index,
                                       num_decoder_tokens,
                                       reverse_target_char_index,
                                       max_decoder_seq_length)
    K.clear_session()
    return decoded_sentence

    
def convert_english_to_french(APP_ROOT, text):
    import urllib.parse
    text = urllib.parse.unquote_plus(text)
    # prepare encoder model
    workingFolder = os.path.join(APP_ROOT,'models','nlp','Sequence to Sequence')
#    model = load_model(os.path.join(workingFolder,'model_trained.h5'))
    encoder_model = load_model(os.path.join(workingFolder,'encoder_model.h5'))
    decoder_model = load_model(os.path.join(workingFolder,'decoder_model.h5'))
    
    #import json
    #final_dictionary = json.loads(encoder_model.to_json()) 
    #json_str = decoder_model.to_json()
    
    input_token_index = Pickle_LoadObject(os.path.join(workingFolder, 'input_token_index.pickle'))
    num_encoder_tokens = len(input_token_index)
    
    reverse_target_char_index = Pickle_LoadObject(os.path.join(workingFolder, 'reverse_target_char_index.pickle'))
    target_token_index = Pickle_LoadObject(os.path.join(workingFolder, 'target_token_index.pickle'))
    num_decoder_tokens = len(target_token_index)
    
    # Some predefine countes based on model
    max_encoder_seq_length = 23
    max_decoder_seq_length = 74
    
    input_texts = []
    input_texts.append(text)
    
    encoder_input_data = np.zeros((len(input_texts),
                                  max_encoder_seq_length,
                                  num_encoder_tokens),
                                  dtype='float32')
    
    # preparing the training data
    for i, input_text in enumerate(input_texts):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        
        # set the remaining character as space so set the index of ' '
        encoder_input_data[i,t+1:,input_token_index[' ']] = 1.
    
    input_seq = encoder_input_data[0:1]
    
    #Encode the input as state vector
    state_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    
    # Populate the first character of target sequence with the start character
    target_seq[0, 0, target_token_index['\t']] = 1.
    
    # Sampling loop for batch sequence
    # to simplify, here we assume a batch size of 1
    stop_condition = False
    decoded_santence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + state_value)
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_santence += sampled_char
        
        # Exit condition: either hit max length
        # or find the stop character
        if (sampled_char == '\n' or len(decoded_santence) > max_decoder_seq_length):
            stop_condition = True
        
        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        
        #update the states
        state_value = [h, c]
    
    K.clear_session()
    return decoded_santence
#    return 'ok',200