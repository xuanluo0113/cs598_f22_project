import glob
import pandas as pd
import json
import datetime
import numpy as np
from icd9_ontology import SecondLevelCodes
import collections

class Dataset(object):

    def __init__(self):
        self.icd_hierarchy = 'codes_2L.json'
        self.words_count = 0
        self.min_freq = 5
        self.max_len_visit = 0
        self.vocabulary_size = 0
        self.digit3_size = 0

    def load_data(self):

        with open('processed/patients_mimic3_full.json') as read_file:
            patients = json.load(read_file)

        total_visits = 0
        all_codes = []  # store all diagnosis codes
        all_cpt_codes = []

        for patient in patients:
            for visit in patient['visits']:
                total_visits += 1
                dxs = visit['DXs']
                for dx in dxs:
                    all_codes.append('D_' + dx)

        # store all codes and corresponding counts
        count_org = []
        count_org.extend(collections.Counter(all_codes).most_common())


        # store filtering codes and counts
        count = []
        for word, c in count_org:
            word_tuple = [word, c]
            if c >= self.min_freq:
                count.append(word_tuple)
                self.words_count += c

        code_no_per_visit = self.words_count / total_visits
                
        dictionary = dict()
        dictionary_3digit = dict()
        code_to_second_level_code_dict = dict()
        slc = SecondLevelCodes(self.icd_hierarchy)
        second_level_codes = []
        # add padding
        dictionary['PAD'] = 0
        for word, cnt in count:
            index = len(dictionary)
            dictionary[word] = index
            if word[:2] == 'D_':
                digit = slc.second_level_codes_icd9(word[2:])
                second_level_codes.append(digit)
                code_to_second_level_code_dict[word] = digit


        self.vocabulary_size = len(dictionary)


        count_second_level_codes = []
        count_second_level_codes.extend(collections.Counter(second_level_codes).most_common())
        for word, cnt in count_second_level_codes:
            index = len(dictionary_3digit)
            dictionary_3digit[word] = index
        self.digit3_size = len(dictionary_3digit)
        print('Number of second level category: ', len(dictionary_3digit))

        self.max_len_visit = 0
        max_visits = 0
         # encoding patient using index
        for patient in patients:
            visits = patient['visits']
            len_visits = len(visits)
            if len_visits > max_visits:
                max_visits = len_visits
            for visit in visits:
                dxs = visit['DXs']
                if len(dxs) == 0:
                    continue
                else:
                    visit['DXs'] = [dictionary['D_' + dx] for dx in dxs if 'D_' + dx in dictionary]
                len_current_visit = len(visit['DXs'])
                if len_current_visit > self.max_len_visit:
                    self.max_len_visit = len_current_visit

        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        print('Length of reverse dictionary ', len(reverse_dictionary))


        # In[20]:


        batches = []
        valid_p_ct = 0
        for patient in patients:
            pid = patient['pid']
            # get patient's visits
            visits = patient['visits']
            # sorting visits by admission date
            sorted_visits = sorted(visits, key=lambda visit: visit['admsn_dt'])
            
            # get valid visit
            valid_visits = []
            for v in sorted_visits:
                if len(v['DXs']) > 0:
                    valid_visits.append(v)
            no_visits = len(valid_visits)
            if no_visits < 2:
                continue
            valid_p_ct += 1

            # only use last 10 visits if number of visits is larger than 11
            max_visit_visit = 10

            last_visit = valid_visits[no_visits - 1]
            second_last_visit = valid_visits[no_visits - 2]
            # third_last_visit = valid_visits[no_visits - 3]


            ls_codes = []
            ls_intervals = []
            if no_visits > max_visit_visit + 1:
                feature_visits = valid_visits[no_visits - (max_visit_visit + 1) : no_visits - 1]
            else:
                feature_visits = valid_visits[:no_visits - 1]

            n_visits = len(feature_visits)

            first_valid_visit_dt = datetime.datetime.strptime(feature_visits[0]['admsn_dt'], "%Y%m%d")
            for i in range(n_visits):
                visit = feature_visits[i]
                codes = visit['DXs']

                if len(codes) == 0:
                    n_zeros += 1

                current_dt = datetime.datetime.strptime(visit['admsn_dt'], "%Y%m%d")
                interval = (current_dt - first_valid_visit_dt).days + 1
                ls_intervals.append(interval)
                code_size = len(codes)
                
                # code padding
                if code_size < self.max_len_visit:
                    list_zeros = [0] * (self.max_len_visit - code_size)
                    codes.extend(list_zeros)
                ls_codes.append(codes)
                
            # visit padding
            if n_visits < max_visit_visit:
                for i in range(max_visit_visit - n_visits):
                    list_zeros = [0] * self.max_len_visit
                    ls_codes.append(list_zeros)
                    ls_intervals.append(0)

            # --------- readmission label --------------------
            last_dt = datetime.datetime.strptime(last_visit['admsn_dt'], "%Y%m%d")
            second_last_dt = datetime.datetime.strptime(second_last_visit['admsn_dt'], "%Y%m%d")
            # thrid_last_dt = datetime.datetime.strptime(third_last_visit['admsn_dt'], "%Y%m%d")
            days = (last_dt - second_last_dt).days
            
            if days <= 30:
                adm_label = 1
            else:
                adm_label = 0
            # print(days, adm_label)
            # --------- diagnosis label --------------------
            one_hot_labels = np.zeros(len(dictionary_3digit)).astype(int)
            last_codes = last_visit['DXs']
            # print(last_codes)
            for code in last_codes:
                if int(code) in reverse_dictionary:
                    code_str = reverse_dictionary[int(code)]
                    if code_str in code_to_second_level_code_dict:
                        cat_code = code_to_second_level_code_dict[code_str]
                        index = dictionary_3digit[cat_code]
                    # else:
                    #     print(code_str)
                    #     index = 0
                one_hot_labels[index] = 1

            batches.append(
                [pid, ls_intervals, ls_codes, one_hot_labels, adm_label])
            
        # print('total number of valid patient', valid_p_ct)
        codes = []
        dx_labels = []
        re_labels = []
        pids = []
        intervals = []
        for batch in batches:
            pids.append(batch[0])
            intervals.append(batch[1])
            codes.append(batch[2])
            dx_labels.append(batch[3])
            re_labels.append(batch[4])

        return batches
