import torch
import pickle

name = {'xEffect':'This person','xReact':'This person feels','oEffect':'Others','oReact':'Others feel'}

def processed_sep(utter_know,value):
    source = utter_know[value].split(' ==sep== ')
    if ' none' in source:
        source.remove(' none')
    rlt = name[value]+source[0]
    return rlt

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'
#原始对话
train_data_path = '/home/DATA1/gxj/bare-RUN/dd_data/dailydialog_' + TRAIN + '.pkl'
dev_data_path = '/home/DATA1/gxj/bare-RUN/dd_data/dailydialog_' + DEV + '.pkl'
test_data_path = '/home/DATA1/gxj/bare-RUN/dd_data/dailydialog_' + TEST + '.pkl'
iemocap_data_path = '/home/data1/gxj/iemocap_test.pkl'
#原始CSK
train_knowledge_path = '/home/DATA1/gxj/bare-RUN/dd_data/dailydialog_'+TRAIN+'_know.pkl'
dev_knowledge_path = '/home/DATA1/gxj/bare-RUN/dd_data/dailydialog_'+DEV+'_know.pkl'
test_knowledge_path = '/home/DATA1/gxj/bare-RUN/dd_data/dailydialog_'+TEST+'_know.pkl'
iemocap_knowledge_path = '/home/data1/gxj/iemocap_test_know.pkl'

#经过处理的CSK
train_processed_knowledge_path = '/home/DATA1/gxj/bare-RUN/dd_data/knowledge_'+TRAIN+'.pkl'
dev_processed_knowledge_path = '/home/DATA1/gxj/bare-RUN/dd_data/knowledge_'+DEV+'.pkl'
test_processed_knowledge_path = '/home/DATA1/gxj/bare-RUN/dd_data/knowledge_'+TEST+'.pkl'
iemocap_processed_knowledge_path = '/home/data1/gxj/iemocap_knowledge.pkl'
#对话+CSk
base_know_train_path = '/home/DATA1/gxj/bare-RUN/dd_data/base_knowledge_'+TRAIN+'.pkl'
base_know_dev_path = '/home/DATA1/gxj/bare-RUN/dd_data/base_knowledge_'+DEV+'.pkl'
base_know_test_path = '/home/DATA1/gxj/bare-RUN/dd_data/base_knowledge_'+TEST+'.pkl'
base_know_iemocap_path = '/home/data1/gxj/base_knowledge_iemocap.pkl'

#处理CSK
def knowledge_processed(knowledge_path,processed_knowledge_path):
    knowledge = pickle.load(open(knowledge_path, 'rb'), encoding='latin1')
    all_conversation_know_processed = []
    for conv_id in range(len(knowledge)):
        conversation_know = knowledge[conv_id]
        conversation_know_processed = []
        
        for utter_id in range(len(conversation_know)):
            utter_know = conversation_know[utter_id]
            source_xEffect = processed_sep(utter_know,'xEffect')+'.'
            source_xReact = processed_sep(utter_know,'xReact')+'.'
            source_oEffect = processed_sep(utter_know,'oEffect')+'.'
            source_oReact = processed_sep(utter_know,'oReact')+'.'
            source = source_xEffect+source_xReact+source_oEffect+source_oReact
            conversation_know_processed.append(source)
            
        all_conversation_know_processed.append(conversation_know_processed)
    
    pickle.dump(all_conversation_know_processed, open(processed_knowledge_path, 'wb'))

# knowledge_processed(train_knowledge_path,train_processed_knowledge_path)
# knowledge_processed(dev_knowledge_path,dev_processed_knowledge_path)
# knowledge_processed(test_knowledge_path,test_processed_knowledge_path)
# knowledge_processed(iemocap_knowledge_path,iemocap_processed_knowledge_path)

#将CSK添加到对话中
def Add_know(processed_knowledge_path,data_path,base_know_path):
    know = pickle.load(open(processed_knowledge_path, 'rb'), encoding='latin1')
    data = pickle.load(open(data_path, 'rb'), encoding='latin1')

    for i in range(len(data)):
        conv_len = len(data[i])
        for j in range(conv_len):
            data[i][j]['knowledge']=know[i][j]

    pickle.dump(data, open(base_know_path, 'wb'))
    
# Add_know(train_processed_knowledge_path,train_data_path,base_know_train_path)
# Add_know(dev_processed_knowledge_path,dev_data_path,base_know_dev_path)
# Add_know(test_processed_knowledge_path,test_data_path,base_know_test_path)
Add_know(iemocap_processed_knowledge_path,iemocap_data_path,base_know_iemocap_path)



#test
# train_base_know = pickle.load(open(base_know_train_path, 'rb'), encoding='latin1')
# dev_base_know = pickle.load(open(base_know_dev_path, 'rb'), encoding='latin1')
# test_base_know = pickle.load(open(base_know_test_path, 'rb'), encoding='latin1')
base_know_iemocap = pickle.load(open(base_know_iemocap_path, 'rb'), encoding='latin1')

# print(len(train_base_know))
# print(train_base_know[0][0])

# print(len(dev_base_know))
# print(dev_base_know[0][0])

# print(len(test_base_know))
# print(test_base_know[0][0])

print(len(base_know_iemocap[0]))
for i in range(len(base_know_iemocap[0])):
    print(base_know_iemocap[0][i])


    
    
          