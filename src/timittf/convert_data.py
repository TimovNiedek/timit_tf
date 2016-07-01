# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:18:43 2016

@author: Timo van Niedek

This module can be used for converting mfcc and phone data to pickles that can
be used as input for the DNN module.

"""

import numpy as _np
import os as _os
import tarfile as _tarfile
import cPickle as _cPickle
import argparse

phone_classes = None

def match_data(data_path, mfcc_file):
    missing_count = 0
    disc_count = 0
    total_count = 0    
    discrepancy = 0
    
    data = []
    labels = []
    phone_times = []    
    
    with _tarfile.open(mfcc_file) as tar:
        for root, dirs, files in _os.walk(data_path):
            for file in files:
                if file.endswith(".phn"):
                    total_count += 1                    
                    
                    # Code of the speaker
                    sp = file[3:8]
                    # Code of the sentence collection
                    sc = file[8:10]
                    
                    # Skip the "sa" sentences
                    if (sc == "sa"):
                        continue
                    
                    # Remove the extension from the file
                    base = _os.path.splitext(file)[0]
                    # The mfcc's name in the tar is txt/dr_base.txt
                    mfcc_name = "txt/" + sp + "_" + base + ".txt"           
                    try:
                        mfcc = tar.getmember(mfcc_name)
                        mfcc_data = read_mfcc(tar.extractfile(mfcc))
                        phones, mfcc_data, phn_times, d = read_phn(root+"/"+file, mfcc_data)
                        if (d < -5):
                            print file, "has discrepancy of", d
                        if (d > 5):
                            print file, "has discrepancy of", d
                        if (d != 0):
                            disc_count += 1
                        discrepancy += d
                        data.append(mfcc_data)
                        labels.append(phones)
                        phone_times.append(phn_times)
                    except KeyError:
                        # Do nothing, the file does not exist in the tar
                        missing_count += 1
                        pass
                    if total_count % 100 == 0:
                        print total_count, " files parsed"                    
    
    print "Total mfcc's found:", total_count
    print "Missing .phn files:", missing_count
    print "Cases with discrepancy:", disc_count
    print "Average discrepancy:", (float(discrepancy) / float(total_count-missing_count))
    return data, labels, phone_times

def convert_times((start, end, phone)):
    return (int(start/160.0), int(end/160.0), phone)

def read_phn(f, temp_mfcc):
    temp_phones = _np.loadtxt(f, dtype={'names':('start', 'end', 'phone'), 
                               'formats':(_np.int32, _np.int32, 'S4')})
    
    # Get the length of the phone data
    _, phn_len, _ = temp_phones[-1]    
    phn_len_mill = int(phn_len/160)
    
    if phn_len_mill < len(temp_mfcc):
        # An array of class labels for every 10 ms in the phone data
        # phones[2] is the phoneme annotated from 20-30 ms
        phones = _np.empty(phn_len_mill, dtype=int)
        
        # Make sure the length of mfcc_data and phn_len_mill are equal
        mfcc_data = temp_mfcc[0:phn_len_mill]
    else:
        phones = _np.empty(len(temp_mfcc))
        mfcc_data = temp_mfcc
    
    d = phn_len_mill - len(temp_mfcc)
        
    # A mask for filtering the q phonemes from the mfcc and phone data
    qfilter = _np.ones(len(phones), dtype=bool)

    # Create an array to store the start and end times of a phone
    phone_times = _np.empty(len(temp_phones), dtype={'names':('start', 'end', 'phone'), 
                               'formats':(_np.int32, _np.int32, 'S4')})
                               
    # Create a seperate q-filter for this array
    qfilter_times = _np.ones(len(phone_times), dtype=bool)    
    
    # If a q phone is found, the times after have to be adjusted accordingly,
    # as if there was never a q-phone.
    qdif = 0
    
    # Convert the string phonemes to class labels
    for i, (s, e, phone) in enumerate(temp_phones):
        start = int(s/160.0)
        end = int(e/160.0)

        phone_times[i] = (start-qdif,end-qdif,phone)        
        
        # Keep track of which phonemes were 'q'
        if phone == 'q':
            qfilter[start:min(end,len(phones))] = False
            qfilter_times[i] = False
            qdif += end-start
        else:
            phones[start:min(end,len(phones))] = _np.where(phone_classes == phone)[0][0]
            
    return phones[qfilter].astype(int), mfcc_data[qfilter], phone_times[qfilter_times], d
    
def read_mfcc(mfcc):
    mfcc_data = _np.loadtxt(mfcc)
    return mfcc_data    

def main(data_folder, target_folder):
    with open(data_folder+'/train/phones.pickle', 'rb') as f:
        global phone_classes
        phn_temp = _np.asarray(_cPickle.load(f))
        phone_classes = _np.delete(phn_temp, _np.where(phn_temp == 'q'))
    
    print "-----------------"
    print "PARSING TEST DATA"
    print "-----------------"
    test_data, test_labels, test_phone_times = match_data(data_folder+"/test", data_folder+"/timit-txt.tar.gz")
    """
    with open(target_folder+'/test_data.pickle', 'w') as ted_dump:
        _cPickle.dump(test_data, ted_dump)
    with open(target_folder+'/test_labels.pickle', 'w') as tel_dump:
        _cPickle.dump(test_labels, tel_dump)
    with open(target_folder+'/test_times.pickle', 'w') as tet_dump:
        _cPickle.dump(test_phone_times, tet_dump)"""
    
    print "------------------"
    print "PARSING TRAIN DATA"
    print "------------------"
    # We don't need the phone times for the training data
    train_data, train_labels, _ = match_data(data_folder+"/train", data_folder+"/timit-txt.tar.gz")
    """
    with open(target_folder+'/train_data.pickle', 'w') as trd_dump:
        _cPickle.dump(train_data, trd_dump)
    with open(target_folder+'/train_labels.pickle', 'w') as trl_dump:
        _cPickle.dump(train_labels, trl_dump)
        """
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Convert the TIMIT dataset to pickles.')
    ap.add_argument('data_folder', type=str, help='Folder containing the TIMIT dataset (should contain folders test & train and file timit-txt.tar.gz)')    
    ap.add_argument('target_folder', type=str, help='Target folder for pickle files.')    
    args = ap.parse_args()
    main(args.data_folder, args.target_folder)
    
    