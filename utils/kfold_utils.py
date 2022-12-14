# K-fold Cross validation to avoid overfitting to specific subject/labels

import csv
import json

def splitter(data_size):
    arr = [0] * 3
    arr[0] = round(data_size*7/10)
    arr[1] = round(data_size*2/10)
    arr[2] = round(data_size*1/10)
    return arr

def init_data(label_mode):
    '''
        initialize subjDict and csvDict data by reading the entire dataset
        subjDict: dictionary of true_labels mapped to sample count
        csvDict: dictionary of subject-true_label pair mapped to full csv row 
                    (in a format writable to data indice txt file)
    '''
    PATH = '/home/johnh/iccad-tinyml/data_indices/total_indice.csv'
    csvf = PATH
    # subject-true_label => writable full csv row
    csvDict = {}
    # subject => true_label => sample count
    subjDict = {}
    # for SRVT mode splitting 
    total_pos_cnt = 0
    total_neg_cnt = 0
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            binary_label = row[0]
            filename = row[1]
            seg = filename.split("-")
            subject = seg[0] # S01
            true_label = seg[1] # AFb

            if label_mode == 'SRVT' and true_label not in ['SR', 'VT']:
                continue
            # update csvDict
            subject_label = subject + "-" + true_label
            if subject_label not in csvDict.keys():
                csvDict[subject_label] = []
            csvRow = str(binary_label) + "," + filename
            csvDict[subject_label].append(csvRow)
            # update subjDict
            if subject not in subjDict.keys():
                subjDict[subject] = {}
            if true_label not in subjDict[subject].keys():
                subjDict[subject][true_label] = 0
            subjDict[subject][true_label] += 1

            if label_mode == 'SRVT':
                if true_label == 'VT':
                    total_pos_cnt += 1
                else:
                    total_neg_cnt += 1
    # all 0 if not SRVT
    pos_split = splitter(total_pos_cnt)
    neg_split = splitter(total_neg_cnt)
    tot = sum(pos_split) + sum(neg_split)
    return subjDict, csvDict, pos_split, neg_split, tot
    
def subj_load_balancer(subj_list, subj_Dict):
    balanced_subj_list = []
    # loop the unselected subject list, add to the posCnt and negCnt, take the diff. add the subject that
    # minimizes the diff. 
    posCnt = 0
    negCnt = 0
    while len(subj_list) != 0:
        minSubj = ''
        gap = 1000000
        updatePosCnt = 0
        updateNegCnt = 0
        for subj in subj_list:
            
            # print(subj_Dict[subj]["SR"])
            tempPosCnt = posCnt + subj_Dict[subj]["VT"]
            tempNegCnt = negCnt + subj_Dict[subj]["SR"]
            curGap = abs(tempPosCnt-tempNegCnt)
            # print(f" {tempPosCnt}, {tempNegCnt}")
            if curGap < gap:
                gap = curGap
                minSubj = subj
                updatePosCnt = tempPosCnt
                updateNegCnt = tempNegCnt
        # print(f"gap: {gap}")
        
        posCnt = updatePosCnt
        negCnt = updateNegCnt

        balanced_subj_list.append(minSubj)
        subj_list.remove(minSubj)
        # print(balanced_subj_list, gap)
        # print(len(balanced_subj_list))
        # print(len(subj_list))
    return balanced_subj_list

def data_split(subjDict, subj_list, subj_offset, split_threshold, subj_split_dist):
    '''
        warning: currently only works for SRVT combo
        naively split the dataset into train, val, test sets according to the 7:2:1 ratio.
        profile the sample size of actual data split distribution as well
    '''
    # list storing subject-ground_label pair 
    train_list = []
    val_list = []
    test_list = []
    # dictionary of the actual data split distribution
    real_split = {}
    real_split['VT'] = {'train':0,'val':0,'test':0}
    real_split['SR'] = {'train':0,'val':0,'test':0}
    
    VTcnt = 0
    SRcnt = 0

    label_list = ["VT", "SR"]

    for idx in range(len(subj_list)):
        subject = subj_list[(idx+subj_offset)%len(subj_list)]
        SR_VT_dict = subjDict[subject]
        # which split to assign the pair depends on VT or SR count
        SRcnt += SR_VT_dict["SR"]
        newCnt = VTcnt + SR_VT_dict["VT"]
        # assign the appropriate split to the pair
        if newCnt < split_threshold[0]:
            mode = 'train'
            mode_list = train_list
        elif newCnt < split_threshold[0]+split_threshold[1]:
            mode = 'val'
            mode_list = val_list
            # print(f"{subject} has VT: {subjDict[subject]["VT"]}")
            # print(f"{subject} has SR: {subjDict[subject]["SR"]}")
        else:
            mode = 'test'
            mode_list = test_list
        
        subj_split_dist[mode].add(subject)

        for label in label_list:
            subject_label = subject + '-' + label
            if SR_VT_dict[label] != 0:
                mode_list.append(subject_label)
            real_split[label][mode] += SR_VT_dict[label]
        VTcnt = newCnt
    return train_list,val_list,test_list,real_split,subj_split_dist 

def write_to_csv(train_list, val_list, test_list, csvDict):
    # write to data_indices files
    SAVE_PATH = './data_indices/'
    for mode in ['train','val','test']:
        with open(SAVE_PATH+'subj_split_{}_indice.csv'.format('total'), 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            if mode == 'train':
                csv_list = train_list
            elif mode == 'val':
                csv_list = val_list
            elif mode == 'test':
                csv_list = test_list
            for subject_label in csv_list:
                for csvRow in csvDict[subject_label]:
                    writer.writerow([csvRow])
