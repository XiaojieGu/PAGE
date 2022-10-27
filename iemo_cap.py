'''
Author: RenzeLou marionojump0722@gmail.com
Date: 2022-09-14 22:11:12
LastEditors: RenzeLou marionojump0722@gmail.com
LastEditTime: 2022-09-15 21:55:20
FilePath: /7-bare-RUN/iemo_cap.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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
    # print(all_cap)   
    most_cap.append(all_cap)
    
# pickle.dump(most_cap, open(processed_data_path, 'wb'))

data = pickle.load(open(processed_data_path, 'rb'), encoding='latin1')
print(data[0])