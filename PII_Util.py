import json
import os
import pandas as pd
import re

import spacy
from spacy import displacy
from spacy.tokens import Span, Doc

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import numpy as np
from sklearn.metrics import fbeta_score

# import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
from transformers import pipeline
import torch

from collections import Counter

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import functools

# Dataset Specs

classes = ['O',
'B-EMAIL',
'B-ID_NUM',
'B-NAME_STUDENT',
'B-PHONE_NUM',
'B-STREET_ADDRESS',
'B-URL_PERSONAL',
'B-USERNAME',
'I-ID_NUM',
'I-NAME_STUDENT',
'I-PHONE_NUM',
'I-STREET_ADDRESS',
'I-URL_PERSONAL',
]


classes2id = {clas:i for i, clas in enumerate(classes)}
id2classes = {i:clas for i, clas in enumerate(classes)}
classes_pos = classes[1:]
classes_pos_id = [classes2id[label] for label in classes_pos]
entities = ['O', 'EMAIL', 'ID_NUM', 'NAME_STUDENT', 'PHONE_NUM', 'STREET_ADDRESS', 'URL_PERSONAL', 'USERNAME']
entity2id = {entity:i for i, entity in enumerate(entities)}

entity2class = {entity:[f'B-{entity}',f'I-{entity}'] for entity in entities}
entity2class['O'] = ['O','O']
entity2class['EMAIL'][1] = entity2class['EMAIL'][0]
# entity2class['ID_NUM'][1] = entity2class['ID_NUM'][0]
entity2class['USERNAME'][1] = entity2class['USERNAME'][0]

entity_id2class_id = {entity2id[entity]:[classes2id[label] for label in classes] for entity,classes in entity2class.items()}
np_entity_id2class_id_b = np.array([class_dict[0] for class_dict  in entity_id2class_id.values()])
np_entity_id2class_id_i = np.array([class_dict[1] for class_dict  in entity_id2class_id.values()])
class_entity_sorted = sorted([(v,k) for k, values in entity_id2class_id.items() for v in values])
class_id2entity_id = {k:v for k,v in  class_entity_sorted}
np_class_id2entity_id = np.array([entity for entity in class_id2entity_id.values()])

# np_class_id2entity_id
# np_entity_id2class_id = np.torch([classes for classes in entity_id2class_id.values()])

# Model Adapters

class PII_Adapter():
    def __init__(self,model_name,  threshold=0.5,checkpoint_path=None, from_scratch = False,
                 config_only = False, will_bio_tokens = True, will_bio_words = True, mode_bio='preserve'):

        self.mode_bio = mode_bio

        if mode_bio not in ['preserve','reorder']:
            raise Exception('invalid value for mode_bio')
         
        # ----------- Model Initialization
        self.threshold = threshold
        
        self.model_name = model_name
        self.tokenizer_name = model_name

        if checkpoint_path:
            self.model_name = checkpoint_path

        if checkpoint_path or from_scratch:

            if mode_bio == 'preserve':
                #Same labels as classes
                self.cur_label2model_id = classes2id
                
            elif mode_bio == 'reorder':
                #classes + I- labels outside of classes
                self.cur_label2model_id, self.np_model_id2cur_entity_id = get_subword_class_mappings(classes2id)


            num_labels = len(self.cur_label2model_id)

        #Instantiate Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        #Instantiate Model and Config
        if config_only:

            #For cross compatibility
            if checkpoint_path:
                label2id = self.cur_label2model_id
                id2label = {i:label for label,i in label2id.items()}
                
                self.config = AutoConfig.from_pretrained(self.model_name,id2label = id2label, label2id = label2id)

            else:
                self.config = AutoConfig.from_pretrained(self.model_name)
            
        else:
            if num_labels:

                label2id = self.cur_label2model_id
                id2label = {i:label for label,i in label2id.items()}
                
                config = AutoConfig.from_pretrained(self.model_name, num_labels=num_labels, id2label = id2label, label2id = label2id)
                self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, config=config,  ignore_mismatched_sizes=True)
                
            
            else:
                self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)

            self.config = self.model.config

        self.O_id = self.config.label2id['O']
        
        # ----------- Metrics, etc.

        self.labels = classes
        self.label_pos_ids = classes_pos_id
        
#         self.has_diff_labels = False

        if getattr(self, 'np_model_id2cur_entity_id', None) is None:
            self.np_model_id2cur_entity_id = np_class_id2entity_id
                
        self.will_bio_tokens = will_bio_tokens
        self.will_bio_words = will_bio_words
                
        if will_bio_tokens:
            # self.n_token_labels = len(classes) # Old
            self.n_token_labels = len(self.config.label2id)
            
        else:
            self.n_token_labels = len(self.config.label2id) 

#Model not in BIO Format
class Yanis_Adapter(PII_Adapter):
    def __init__(self, threshold = 0.1, checkpoint_path = None, from_scratch = False, 
                 config_only = False,  will_bio_tokens = True, will_bio_words = True, mode_bio='preserve'):
        
        super().__init__("Yanis/microsoft-deberta-v3-large_ner_conll2003-anonimization_TRY_1", threshold=threshold, checkpoint_path=checkpoint_path, from_scratch=from_scratch, config_only=config_only, 
                         will_bio_tokens = will_bio_tokens, will_bio_words = will_bio_words, mode_bio=mode_bio)
            
        #If not fine-tuned
        if not checkpoint_path and not from_scratch:
#            self.will_bio_tokens = True # Override
#            self.has_diff_labels = True
            self.O_id = self.config.label2id['O']
            self.model_id2cur_ent_label = {
                0:'O', 
                1:'O',
                2:'NAME_STUDENT',
                3:'O',
                4:'PHONE_NUM',
                5:'O',
                6:'O',
                7:'O',
                8:'ID_NUM',
                9:'O',
                10:'ID_NUM',
                11:'O',
                12:'STREET_ADDRESS',
                13:'O',
                14:'EMAIL',
                15:'O',
                16:'O',
                17:'O'}

            
            self.labels = [label for label in self.config.label2id.keys()]
            self.label_pos_ids = [i for i,label in self.model_id2cur_ent_label.items() if label!='O']
            self.model_id2cur_entity_id = {model_id: entity2id[self.model_id2cur_ent_label[model_id]] for model_id,_ in self.model_id2cur_ent_label.items()}
            self.np_model_id2cur_entity_id = np.array([entity2id[self.model_id2cur_ent_label[model_id]] for model_id,_ in self.model_id2cur_ent_label.items()], dtype='int8')
            #B Only
    #         self.model_id2cur_id = {key:classes2id[value] for key,value in self.model_id2cur_label.items()}

            self.labels_irrelevant = [key for key,value in self.model_id2cur_ent_label.items() if value == 'O' and key != self.O_id]

            rev_model_id2cur_ent_label = {value:key for key,value in  self.model_id2cur_ent_label.items()}
            rev_model_id2cur_ent_label['O'] = self.O_id

            cur_label2model_id = {'O': self.O_id}
            for label in classes:
                if label == 'O':  
                    continue
                else:
                    entity = label.split('-')[1]

                    if entity in rev_model_id2cur_ent_label:
                        cur_label2model_id[label] = rev_model_id2cur_ent_label[entity]        
                    else:
                        cur_label2model_id[label] = rev_model_id2cur_ent_label['O']

            self.cur_label2model_id = cur_label2model_id


def get_subword_class_mappings(class2id):

    ents_with_i = set([class_str.split('-')[1] for class_str, i in class2id.items() if len(class_str.split('-')) == 2 and class_str.split('-')[0] == 'I'])
    # entities = set([class_str.split('-')[1] for class_str, i in class2id.items() if len(class_str.split('-')) == 2])

    #Exclude O with [1:]
    ents_no_i = set(entities[1:]) - ents_with_i

    subword_class2id = class2id.copy()
    model_id2cur_entity_id = np_class_id2entity_id.tolist()
    
    for ent in ents_no_i:
        new_class = f'I-{ent}'
        class_id = len(subword_class2id)
        
        subword_class2id[new_class] = class_id
        model_id2cur_entity_id.append(entity2id[ent])

    np_model_id2cur_entity_id = np.array(model_id2cur_entity_id)


    return subword_class2id, np_model_id2cur_entity_id


#---------- Split

def stratified_split(df_unsplitted, possible_labels, test_size=0.12, random_state=42):
    df_dist = label_count_columns(df_unsplitted, classes)
    df_dist = perform_dbscan_clustering(df_dist, eps=0.5, min_samples=5)

    df_train, df_valid = train_test_split(df_unsplitted, test_size=test_size, random_state=random_state, stratify=df_dist['cluster'])
    return df_train, df_valid

def label_count_columns(df, possible_labels):
    # Initialize a dictionary to store the count of each label
    label_counts = {label: [] for label in possible_labels}
    
    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Use Counter to count the occurrence of each label in the sequence
        row_label_counts = Counter(row['labels'])
        
        # Append the counts to the label_counts dictionary
        for label in possible_labels:
            label_counts[label].append(row_label_counts[label])
    
    # Create a new DataFrame from the label_counts dictionary
    label_counts_df = pd.DataFrame(label_counts)
    
    return label_counts_df


def perform_dbscan_clustering(label_counts_df, eps=0.5, min_samples=5):
    # Standardize the feature values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(label_counts_df)
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(scaled_features)
    
    # Add cluster labels to the DataFrame
    label_counts_df['cluster'] = dbscan.labels_
    
    return label_counts_df


#---------- Preprocessing 

def preprocess_from_words(example, model_adapter, col_words = 'tokens', col_labels = "labels", mode_bio = "preserve"):
    
    tokenizer = model_adapter.tokenizer
    cur_label2model_id = getattr(model_adapter, 'cur_label2model_id', None) 
    
    #Encode
    inputs = tokenizer(example[col_words], return_tensors='pt', return_offsets_mapping=False, is_split_into_words=True)
    tokens = inputs.tokens()
    word_ids = inputs.word_ids()

    #Labels adjusted (words to subwords)
    labels = words_bio2subwords_bio(word_ids, example[col_labels], mode_bio=mode_bio)

    #Dataset Label to Model ID
    if cur_label2model_id:
        label_ids = [cur_label2model_id[label] for label in labels]
        
    else: 
        label_ids = [classes2id[label] for label in labels]
        
    labels_aligned = labels

    #To Dataset/Dict
    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "tokens": tokens,
        "token_labels":labels,
        "words": example[col_words],
        "word_labels": example[col_labels],
        "word_ids": inputs.word_ids(),
        "labels": torch.tensor(label_ids),
    }


def hf_offset_mapper(tokenizer):

    def mapper(text):
        token_infos = tokenizer(cur_text, return_offsets_mapping=True, add_special_tokens=False)
        token_infos['start'], token_infos['end'] = zip(*token_infos['offset_mapping'])
        token_infos['tokens'] = tokenizer.tokenize(text, add_special_tokens=False)
        return token_infos
        
    return mapper


def spacy_offset_mapper(nlp):

    def mapper(text):
        
        doc = nlp(text)

        tokens = [token_info.text for token_info in doc]
        start = [token_info.idx for token_info in doc]
        end = [token_info.idx+len(token_info.text) for token_info in doc]

        return {'tokens':tokens,
                'start':start,
                'end':end}

    return mapper
            
def attach_prefix(label, is_beginning = True):
    
    if is_beginning:
        return f'B-{label}'
        
    else:
        return f'I-{label}'

def catch_return(return_value=pd.NA):
    def wrapper(func):
        def wrapped_function(row, *args, **kwargs):
            try:
                return func(row, *args, **kwargs)
            except Exception as e:
                print(f"An error occurred at index {row.name}: {e}")
                return return_value
        return wrapped_function
    return wrapper


def spans_to_tokens(row, offset_mapper, return_bio = True):

    global token_infos, token_labels
    
    cur_text = row.source_text
    span_infos = row.privacy_mask 

    token_infos = offset_mapper(cur_text)
    
    token_labels = ['O']*len(token_infos['start'])
    token_labels_bio = ['O']*len(token_infos['start'])

    i_token = 0
    
    for span_info in span_infos:
        skip_span = False
        is_beginning = True

        #Update latest token

        #Skip condition ('C1') 
        while token_infos['end'][i_token] < span_info['start']:
            i_token += 1

        #condition('C2')
        #Handle leading whitespace at postprocessing (During token to words -> remove labels from whitespace)
        ##Example: Token entry 0, token 52  (might be important to model (_ -> ' ')
        while token_infos['start'][i_token] < span_info['start']:

            #('C2.5')
            if token_infos['end'][i_token] > span_info['end']:
                print('C2.5')
                i_token += 1
                skip_span = True

            #('C2')
            else:
                token_labels[i_token] = span_info['label']
                token_labels_bio[i_token] = attach_prefix(span_info['label'], is_beginning)
                i_token += 1
                is_beginning = False

        if skip_span:
            continue
            
        #Condition ('C3') - assign label
        # C3 start: token_info['start'] >= span_info['start']
        # C3 End:  token_info['end'] <= span_info['end']
        # print(i_token)  # Debug
        
        
        while i_token < len(token_labels) and token_infos['start'][i_token] >= span_info['start'] and token_infos['end'][i_token] <= span_info['end']:
            token_labels[i_token] = span_info['label']
            token_labels_bio[i_token] = attach_prefix(span_info['label'], is_beginning)
            i_token +=1
            is_beginning = False
            
            
    if return_bio:
        return token_infos['tokens'], token_labels_bio
    
    else:
        return token_infos['tokens'], token_labels



def words_bio2subwords_bio(word_ids, labels, mode_bio='preserve'):
    if mode_bio == 'preserve':
        labels = words_bio2subwords_bio_preserve(word_ids, labels)

    elif mode_bio == 'reorder':
        labels = words_bio2subwords_bio_reorder(word_ids, labels)

    else:
        raise Exception("invalid bio_mode")

    return labels


def words_bio2subwords_bio_reorder(word_ids, labels):
    
    #word to tokens
    token_labels = []
    prev_entity = None
    prev_ent_type = None
    
    # Step 5: Iterate through pairs of words and subwords to count the majority label
    for i, word_id in enumerate(word_ids):
        if word_id is None:
            token_labels.append('O')
#             continue
        else:
            
            try:
                token_label = labels[word_id]
            except Exception as e: # Temp error checking
                print(word_id)
                print(len(labels))
                
                raise(e)
                
            #Outside
            if token_label == 'O':
                token_labels.append('O')
                ent_type = 'O'
                
            else:
                prefix, ent_type = token_label.split('-')
            
                #Same entity: B-ent:B-ent, I-ent,I-ent, B-ent,I-ent
                if prev_entity == token_label or f'I-{prev_ent_type}' == token_label:
                    token_labels.append(f'I-{prev_ent_type}')  

                #New Entity: I-ent: B-ent, x-ent1: x-ent2\
                else:
                    token_labels.append(f'B-{ent_type}')

            prev_entity = token_label
            prev_ent_type = ent_type
                
            
    return token_labels


def words_bio2subwords_bio_preserve(word_ids, labels):
    token_labels = [labels[word_id] if word_id else 'O' for word_id in word_ids]
    return token_labels


def convert_to_bio(tokens, labels):
    bio_labels = []
    current_entity = None
    
    for token, label in zip(tokens, labels):
        if label == 'O':
            bio_labels.append(label)
            current_entity = None
        else:
            if current_entity == label:
                bio_labels.append('I-' + label)
            else:
                bio_labels.append('B-' + label)
            current_entity = label
    
    return bio_labels


#---------- Postprocessing 

def preprocess_logits(logits, label):
    np_probs = torch.softmax(logits, axis=-1) 
    
    return np_probs


def pad_lists(lst, pad_value, max_length = None):
    if max_length is None:
        max_length = max(len(sublist) for sublist in lst)
    padded_lst = np.full((len(lst), max_length), pad_value)
    for i, sublist in enumerate(lst):
        padded_lst[i, :len(sublist)] = sublist
    return padded_lst



#Positive class thresholding
def get_tokens_thresholding(np_probs, true_labels_id, threshold, model_adapter):
    
#     np_probs = loaded_predictions
#     true_labels_id = labels_padded
#     threshold = 0.01
    
    # --- Initialize variables
    label2id =  model_adapter.config.label2id
    o_index = label2id['O']
    
    # --- Get indices of highest probabilities
    np_sorted_indices = np.argsort(np_probs)
    np_max_indices = np_sorted_indices[:,:, -1]
    # np_max = np_sorted_indices[np.arange(np_max_indices.shape[0]), np_max_indices]
    np_max_prob = np.take_along_axis(np_probs, np_max_indices[:, :, np.newaxis], axis=2).squeeze()

    
    # --- Get 2nd max probabilities
    np_2nd_max_indices = np_sorted_indices[:, :, -2]
    # np_2nd_max = np_sorted_indices[np.arange(np_sorted_indices.shape[0]), np_max_indices]
    np_2nd_max_prob = np.take_along_axis(np_probs, np_2nd_max_indices[:, :, np.newaxis], axis=2).squeeze()

    # --- Conditional replace (Positive label thresholding)
    
    #Masks
    np_O_mask = np_max_indices == o_index
    np_threshold_mask = np_2nd_max_prob > threshold
    np_replace_mask = np_threshold_mask & np_O_mask

    #Replace
    np_label_ids = np.where(np_replace_mask, np_2nd_max_indices, np_max_indices)

    # Postprocess labels, convert irrelevant labels to 'O'
    if getattr(model_adapter, 'labels_irrelevant', None):
        np_labels_irrelevant = np.array(model_adapter.labels_irrelevant)
        label_ids_mask = np.isin(np_label_ids, np_labels_irrelevant)
        np_label_ids[label_ids_mask] = o_index
        
#     if model_adapter.has_diff_labels:
#         np_label_ids = model_adapter.

    #Convert to BIO, Does not convert orig np_label_ids
    if model_adapter.will_bio_tokens:
        orig_np_label_ids = np_label_ids.copy()
        np_label_ids = to_bio_vect(np_label_ids, model_adapter.np_model_id2cur_entity_id)
        
    else:
        orig_np_label_ids = np_label_ids # Weird logic, replace reference used by flatten instead?

    # Flatten token(subwords) for metrics
    flat_label_ids = np_label_ids.flatten()
    flat_true_labels_id = true_labels_id.flatten()

    # Remove padding preds for metrics
    mask_padding_inv = flat_true_labels_id != -100
    flat_true_labels_id = flat_true_labels_id[mask_padding_inv]
    flat_label_ids = flat_label_ids[mask_padding_inv]
    
    orig_np_label_ids
    
    
    return orig_np_label_ids, flat_true_labels_id, flat_label_ids

# np_label_ids, flat_true_labels_id, flat_label_ids = get_tokens_thresholding(loaded_predictions, labels_padded, threshold, model_adapter)


#Word-token alignment
def get_word_preds(np_pred_tokens, word_ids_padded, model_adapter, preprocessed_dataset):
    row_size = word_ids_padded.shape[1]
    diff_array = np.diff(word_ids_padded[:,:], axis=1, append=row_size)

    #Calculate row boundaries
    non_zero = np.where(diff_array != 0, 1, 0)
    group_ids = np.cumsum(non_zero, axis=1) 
    row_boundaries = np.cumsum(group_ids.max(axis = 1))

    #Calculate splits, split into groups, Truncate > 10
    split_indices =  np.where(diff_array.ravel() != 0)[0] + 1 # Orig
    list_groups = np.split(np_pred_tokens[:,:].ravel(), split_indices)
    list_groups = [group[:10] if len(group) >= 10 else group for group in list_groups]

    #30s Calculate modes per group
    word_labels = [Counter(arr.tolist()).most_common(1)[0][0] if len(arr) > 0 else None for arr in list_groups]

    #Fast (split into rows)
    np_word_labels = np.array(word_labels)
    list_pred_incomp = np.split(np_word_labels, row_boundaries)

    #10s Initialize row arrays for word-level predictions
    list_pred_words = [np.full((len(words)), model_adapter.O_id, dtype='int8') for words in preprocessed_dataset['words']]

    #5s Get word_indices not skipped in word_ids (Handles CLS and Padding tokens)
    list_word_indices = []
    for word_ids in preprocessed_dataset['word_ids']:
        indices = set(word_ids)
        indices.discard(None)
        list_word_indices.append(np.array(list(indices)))

    #200ms Align the processed word-level predictions to initialized array
    for pred_words, word_indices, pred_incomp in zip(list_pred_words, list_word_indices, list_pred_incomp):
        pred_words[word_indices] = pred_incomp[1:-1]
        
    if model_adapter.will_bio_words:
        list_pred_words = [to_bio_vect(pred_words,  model_adapter.np_model_id2cur_entity_id) for pred_words in list_pred_words]
    return list_pred_words

# list_pred_words = get_word_preds(np_label_ids, word_ids_padded, model_adapter)

def to_bio_vect(np_pred_words, np_model_id2cur_entity_id = np_class_id2entity_id):
#     global cumsum, out, zeros, change_indices, accumulate, diff_array, b_mask_indices, i_mask_indices, b_words, i_words, b_words_mapped, i_words_mapped
    
    np_pred_words = np_model_id2cur_entity_id[np_pred_words]

    diff_array = np.diff(np_pred_words, axis=-1, prepend=-1)
    change_indices = np.where(diff_array!=0, 0, 1)
    zeros = change_indices == 0 #reverse of change_indices
    cumsum = np.cumsum(change_indices, axis=-1)
    accumulate = np.maximum.accumulate(np.where(zeros, cumsum, 0), axis=-1)
    out = cumsum - accumulate
    b_mask = out == 0
    b_mask_indices = np.where(b_mask)
    i_mask_indices = np.where(~b_mask)
    
    np_pred_words_mapped = np.empty_like(np_pred_words)
    
    b_words = np_pred_words[b_mask_indices]
    i_words = np_pred_words[i_mask_indices]
    
    b_words_mapped = np_entity_id2class_id_b[b_words]
    i_words_mapped = np_entity_id2class_id_i[i_words]
    
    np_pred_words_mapped[b_mask_indices] = b_words_mapped
    np_pred_words_mapped[i_mask_indices] = i_words_mapped
    
    return np_pred_words_mapped


def count_confusion_matrix(predictions, labels, num_classes):
    """
    Count confusion matrix per class ID.
    
    Args:
    - predictions: NumPy array of predicted class labels (shape: [N]).
    - labels: NumPy array of ground truth class labels (shape: [N]).
    - num_classes: Number of classes.
    
    Returns:
    - TP: NumPy array of True Positives per class ID.
    - TN: NumPy array of True Negatives per class ID.
    - FN: NumPy array of False Negatives per class ID.
    - FP: NumPy array of False Positives per class ID.
    """
    
    global confusion_matrix
    
    
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    np.add.at(confusion_matrix, (labels, predictions), 1)
    
    # True Positives (TP)
    TP = np.diag(confusion_matrix)
    
    # True Negatives (TN)
    TN = np.sum(confusion_matrix) - np.sum(confusion_matrix[np.eye(num_classes, dtype=bool)])
    
    # False Negatives (FN)
    FN = np.sum(confusion_matrix, axis=1) - TP
    
    # False Positives (FP)
    FP = np.sum(confusion_matrix, axis=0) - TP
    
    return confusion_matrix, TP, TN, FN, FP


def compute_metrics_per_class(TP, FN, FP, class_label, beta=5):
    
    tp = TP[class_label]
    fn = FN[class_label]
    fp = FP[class_label]
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0 
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if precision + recall > 0 else 0
    
    metrics = {'precision':precision, 'recall':recall, 'f_beta':f_beta,'tp':tp, 'fp':fp, 'fn':fn}
    
    return metrics

def compute_micro_metrics(predictions, labels, num_classes, pos_labels, prefix='', beta=5):
#     global class_metrics
    
    
    confusion_matrix, TP, TN, FN, FP = count_confusion_matrix(predictions = predictions, labels = labels, num_classes = num_classes)
    
    class_metrics = {f'{prefix}{class_label}':compute_metrics_per_class(TP, FN, FP, class_label, beta=beta) for class_label in range(num_classes)}
#     class_metrics = {key:value for class_label in range(num_classes) for key,value in compute_metrics_per_class(TP, FN, FP, class_label, beta=beta, prefix=prefix).items()}
    
    
    # Calculate total true positives, false positives, and false negatives excluding 'O' label
    total_tp = sum(TP[pos_labels])
    total_fp = sum(FP[pos_labels])
    total_fn = sum(FN[pos_labels])
    
    # Compute total precision excluding 'O' label
    total_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    
    # Compute total recall excluding 'O' label
    total_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    
    # Compute total F-beta score excluding 'O' label
    total_f_beta = (1 + beta**2) * (total_precision * total_recall) / (beta**2 * total_precision + total_recall) if total_precision + total_recall > 0 else 0
    
    total_metrics = {'precision':total_precision, 'recall':total_recall, 'f_beta':total_f_beta,'tp':total_tp, 'fp':total_fp, 'fn':total_fn}
    
    return total_metrics, class_metrics, confusion_matrix



def compute_metrics_base(eval_preds, preprocessed_dataset, model_adapter, threshold = 0.1, return_processed=False):
    
    global flat_label_ids, list_word_true_labels
    
    np_probs, true_labels_id, inputs = eval_preds
    
    #---- Token Preds
    np_label_ids, flat_true_labels_id, flat_label_ids = get_tokens_thresholding(np_probs, true_labels_id, threshold, model_adapter)

    #---- Word Preds

    list_word_ids = [[word_id if word_id is not None else -100 for word_id in word_ids] for word_ids in preprocessed_dataset['word_ids'] ]
    word_ids_padded = pad_lists(list_word_ids, -100,  np_probs.shape[1])
    
    list_pred_words = get_word_preds(np_label_ids, word_ids_padded, model_adapter, preprocessed_dataset)
     
    #Overrides BIO if id already in BIO (BIO -> Entity -> BIO) or (non-BIO -> Entity -> BIO)
    #- implemented inside get_word_preds
    
#     list_pred_words_mapped = [to_bio_vect(pred_words,  model_adapter.np_model_id2cur_entity_id) for pred_words in list_pred_words]
    
    flat_word_label_ids = np.concatenate(list_pred_words)
    
    list_word_true_labels = preprocessed_dataset['word_labels']
    
    flat_word_true_labels_id = np.array([classes2id[word] for list_words in list_word_true_labels for word in list_words], dtype='int8')
    
 
    total_metrics, class_metrics, confusion_matrix = compute_micro_metrics(predictions = flat_label_ids, labels = flat_true_labels_id, 
                                                                       #    num_classes = len(model_adapter.labels), pos_labels=model_adapter.label_pos_ids, beta=5, prefix='token_')
                                                                            num_classes = model_adapter.n_token_labels, pos_labels=model_adapter.label_pos_ids, beta=5, prefix='token_')
    
    total_metrics_w, class_metrics_w, confusion_matrix_w = compute_micro_metrics(predictions = flat_word_label_ids, labels = flat_word_true_labels_id, 
                                                                                 num_classes = len(classes), pos_labels=classes_pos_id, beta=5, prefix='word_')
    
    dict_scores = {'token_confusion_matrix' : confusion_matrix,
                  'word_confusion_matrix' : confusion_matrix_w,
                   'token_total_metrics' : total_metrics,
                   'word_total_metrics' : total_metrics_w}
    
    dict_scores.update(class_metrics)
    dict_scores.update(class_metrics_w)

#     dict_scores = {}

    if return_processed:
        dict_processed = {'flat_label_ids' : flat_label_ids,
                  'flat_true_labels_id' : flat_true_labels_id,
                   'flat_word_label_ids' : flat_word_label_ids,
                   'flat_word_true_labels_id' : flat_word_true_labels_id}
        
        return dict_scores, dict_processed
    
    return dict_scores




# ----- Metrics and Comparison


def col_id2label_base(col_name, model_adapter):
    list_name = col_name.split('_')
    
    
    if list_name[-1].isdigit():
        label_id = int(list_name[-1])
        
        if list_name[0] == 'token':
            list_name[-1] = model_adapter.config.id2label[label_id]
        else:
            list_name[-1] = id2classes[label_id]
        return '_'.join(list_name)
        
    else:
        return col_name
    
    
def run_and_compute_metrics(path_dataset, path_preds, model_adapter):
    global df_all_metrics
    
    dataset = load_from_disk(path_dataset)
    np_preds = np.load(path_preds)
    np_labels = pad_lists(dataset['labels'], -100, np_preds.shape[1])
    
    threshold = model_adapter.threshold
    compute_metrics = functools.partial(compute_metrics_base, preprocessed_dataset=dataset, model_adapter=model_adapter, threshold=threshold)
    
    eval_preds = (np_preds, np_labels, None)
    dict_all_scores = compute_metrics(eval_preds)
    
    token_confusion_matrix = dict_all_scores.pop('token_confusion_matrix')
    word_confusion_matrix = dict_all_scores.pop('word_confusion_matrix')
    
    df_all_metrics = pd.DataFrame(dict_all_scores)
    
    col_id2label = functools.partial(col_id2label_base, model_adapter=model_adapter)
    df_all_metrics.columns = df_all_metrics.columns.map(col_id2label)
    df_all_metrics = df_all_metrics.T
    
    return df_all_metrics, token_confusion_matrix, word_confusion_matrix


# ----- Visualizations

def visualize_label(nlp, doc, tokens, labels, options = None):
    start_pos = -1
    span_infos = []
    for label_index, label in enumerate(labels):
        if label!= 'O':
            start_pos = label_index
            end_pos = start_pos + 1
            span_dict = {'start_pos':start_pos, 'end_pos':end_pos, 'label':label}
            span_infos.append(span_dict)

    doc_spans = []
    doc = Doc(nlp.vocab, words=tokens)
    
    for span_info in span_infos:
        _span = Span(doc, span_info['start_pos'], span_info['end_pos'], span_info['label'])
        doc_spans.append(_span)

    doc.spans['sc'] = doc_spans
    displacy.render(doc, style = 'span', options = options)
#     displacy.render(doc, style = 'span')


def visualize_entry(index, dataset, np_preds, vis_mode='words'):

    global word_ids_padded, np_pred_ids, list_pred_words

    #cur_np_probs = np_preds[index]
    cur_entry =  dataset[index]
    cur_labels_id = np.expand_dims(np.array(cur_entry['labels']),axis=0)

    cur_np_probs = np.expand_dims(np_preds[index,:len(cur_entry['labels'])], axis=0)

    np_pred_ids, np_true_label_ids, np_pred_ids_bio = get_tokens_thresholding(cur_np_probs, cur_labels_id, threshold, model_adapter)

    
    nlp = spacy.load("en_core_web_sm") #factor out?


    if vis_mode=='subwords':
        cur_tokens = cur_entry['tokens']
        cur_labels = [model_adapter.config.id2label[token_id] for token_id in np_pred_ids_bio]
        
        doc = Doc(nlp.vocab, words=cur_tokens)
        
        visualize_label(nlp, doc, cur_tokens, cur_labels, options = options_pii)
        
    
    elif vis_mode=='words':

        #---- Word Preds
        word_ids = [word_id if word_id is not None else -100 for word_id in cur_entry['word_ids']]
        word_ids_padded = np.expand_dims(np.array(word_ids), axis=0)

        sub_dataset = dataset.select([index])
        
        list_pred_words = get_word_preds(np_pred_ids, word_ids_padded, model_adapter, sub_dataset)

        cur_tokens = cur_entry['words']
        cur_labels = [model_adapter.config.id2label[token_id] for token_id in list_pred_words[0]]
        
        doc = Doc(nlp.vocab, words=cur_tokens)
        visualize_label(nlp, doc, cur_tokens, cur_labels, options = options_pii)


def visualize_diff_matrix(diff_cm, labels, fmt="d", pos_labels = None, invert_diagonal = True, invert_colors = True, robust=True):
    # Create a copy of the difference matrix to invert the signs of diagonal elements
    
    if invert_diagonal:
        diff_cm_visual = np.copy(diff_cm)
        np.fill_diagonal(diff_cm_visual, -np.diagonal(diff_cm))
    else:
        diff_cm_visual = diff_cm
        
    if pos_labels:
        inverse_labels = np.setdiff1d(list(range(diff_cm.shape[0])), pos_labels)
        
        diff_cm_visual[inverse_labels, :] = 0
        diff_cm_visual[:, inverse_labels] = 0
        
    positive_values = np.where(diff_cm_visual > 0, diff_cm_visual, 0)
    negative_values = np.where(diff_cm_visual < 0, diff_cm_visual, 0)

    # Normalize positive values to range [0, 1]
    max_positive = np.max(positive_values)
    positive_normalized = np.where(positive_values > 0, positive_values / max_positive, 0)

    # Normalize negative values to range [0, -1]
    min_negative = np.min(negative_values)
    negative_normalized = np.where(negative_values < 0, negative_values / min_negative, 0)

    # Combine positive, negative, and zero values
    data_normalized = positive_normalized - negative_normalized
    
    # Set up the color mapping
#     cmap = sns.diverging_palette(150, 10, as_cmap=True)

    if invert_colors:
        cmap = sns.diverging_palette(150, 10, s=90, l=50, as_cmap=True)
    else:
        cmap = sns.diverging_palette(10, 150, s=90, l=50, as_cmap=True)
    
    # Set up the figure
    plt.figure(figsize=(10, 5))
    
    # Plot the heatmap for the difference matrix with specified color mapping
    obj = sns.heatmap(data_normalized, annot=diff_cm, robust=robust, fmt=fmt, cmap=cmap, center=0,
        linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=labels, yticklabels=labels)
    
    plt.title('Confusion Matrix Difference')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    return obj






