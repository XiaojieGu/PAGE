
import json
import pickle
iemocap_path = '/home/DATA1/gxj/iemocap_test.json'
processed_data_path = '/home/DATA1/gxj/iemocap_test.pkl'
f = open(iemocap_path,'r',encoding='utf-8')
m = json.load(f)
most_cap = []


for i in m:
    data = m[i][0]
    all_cap = []

    for utter in data:
        cap = {}
        id = utter['turn']
        cap['id'] = id
        cap['speaker'] = utter['speaker']
        cap['utterance'] = utter['utterance']
        if utter['emotion']=='excited' or utter['emotion']=='happy':
            cap['emotion'] = 'happiness'
        elif utter['emotion']=='sad' or utter['emotion']=='frustrated':
            cap['emotion'] = 'sadness'
        elif utter['emotion']=='angry':
            cap['emotion'] = 'anger'
        elif utter['emotion']=='neutral':
            cap['emotion'] = 'neutral'
        cap['act'] = 'directive'
        if cap['emotion']!= 'neutral' and 'expanded emotion cause evidence' in utter.keys():
            for num in range(len(utter['expanded emotion cause evidence'])):
                if utter['expanded emotion cause evidence'][num]!='b' and utter['expanded emotion cause evidence'][num]>id:
                    utter['expanded emotion cause evidence'][num]='b'
            cap['evidence'] = utter['expanded emotion cause evidence']
        all_cap.append(cap)

    most_cap.append(all_cap)
    

data = pickle.load(open(processed_data_path, 'rb'), encoding='latin1')
