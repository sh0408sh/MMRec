# Data processing
import pandas as pd
import datetime as dt
from dateutil.parser import parse
import re

# Data visulization
import matplotlib as mpl
import matplotlib.pyplot as plt

# read the list until list content is not blnk
def nonblank(list, start) :
    i = start
    while True :

        # condition for end
        if list[i] != '\n' and list[i] != '' :
            break

        # update i
        i = i +1
    return i 

# split list of strings by '\n' or ''
def splitor(list) :
    j = 0 # start point of slicing
    split_data = []

    for i in range(len(list)) :
        if list[i] == '\n' or list[i] =='' :
            if j == i : 
                j = i+1
                continue
            temp = list[j: i]
            split_data.append(temp)
            j = i + 1 

        # update i
        i = i + 1
    
    #print(i,j)
    
    return split_data

# information of server and directory
def process1(list) :
    # information of server and directory

    dic = {}
    
    for i in list :
        a, b = i.split('INFO') # a: time , b : information
        # a = parse(a)

        b = b.replace('██','')
        b = b.replace('\n','')
        b = b.replace(' ','')
        b = b.split(":\t")

        #dic['time'] = a
        dic[b[0]] = b[1]

    df = pd.DataFrame([dic])
    return df   

# informatoin of experiment 
def info_experiment(list) :
    
    temp = {}

    for i in range(len(list)) :

        tem = list[i].replace('\n','')
        if 'INFO' in tem :
            temp['time'] = parse(tem.replace('INFO',''))
        else :
            tem2 = tem.split('=')
            temp[tem2[0]] = tem2[1]
    
    df = pd.DataFrame([temp])

    return df       

# information of data
def data_info(list) :
    dic = {}
    index = [dataset,]

    for i in range(len(list)) :
        tem = list[i].replace('\n','')

        if 'INFO' in tem :
            #temp = l[i].split('INFO')
            #dic3['time'] = [parse(temp[0])]
            pass

        elif '===' in tem :
            temp = tem.replace('=','')
            index.append(temp)

        elif ':' in tem :
            temp = tem.split(": ")
            if temp[0] in dic.keys() :
                #print(dic[temp[0]])
                dic[temp[0]].append(temp[1])
            else :
                dic[temp[0]] = [temp[1]]
        
        else :
            pass

    df = pd.DataFrame(dic, index = index)
    return df

# split exercise data
def splitor_exercise(list) :
    
    ex_data = []

    ex_data.append(list[0]) #information of parameters

    i = 1
    while True :
        if list[i] == ')\n' :
            break
        i = i+1
    ex_data.append(list[2:i]) # information of embedding
    
    ex_data.append(list[i+1]) # number of trainable parameters

    j = i + 2
    while True :
        if '+++++Finished training' in list[i] :
            break
        i = i + 1
    ex_data.append(list[j:i]) # exercise result

    ex_data.append(list[i:i+3]) # best result

    return ex_data 

# parameter information 1
def para_info(string) :
    Parameters = string.replace('\n','').split('Parameters:')[1].replace('=======','').split('=')
    
    rm_word = ['\'','[',']',' ']
    for j in rm_word :
        Parameters[0] = Parameters[0].replace(j,'')
    Parameters[0] = Parameters[0].split(',')

    rm_word2 = [' ','(',')']
    for j in rm_word2 :
        Parameters[1] = Parameters[1].replace(j,'')
    Parameters[1] = Parameters[1].split(',')

    para = {}
    for j in range(len(Parameters[0])) :
        para[Parameters[0][j]] = float(Parameters[1][j])

    df = pd.DataFrame([para])

    return df

# embedding information 
def embedding_info(list) :
    info = {}
    for i in range(len(list)) :
        temp = list[i].replace('\n','')

        temp = temp.split(': ')
        info[temp[0].replace(' ','')] = temp[1]

    df = pd.DataFrame([info])
    return df    

# embedding information 

def num_trainpara(string) :
    dic = {}
    num = re.search(r'\d+', string)
    dic['num of trainable parameters'] = int(num.group(0))
    return pd.DataFrame([dic])

# epoch and time and loss
def etl(state, type, time) : 
    temp = {}

    ex_epoch, ex_time, ex_trainloss = re.findall(r'\d+\.\d+|\d+', state) # experiment epoch, experiment time. experiment train loss

    ex_epoch = int(ex_epoch)
    ex_time = float(ex_time)
    ex_trainloss = float(ex_trainloss)

    temp['type'] = type
    temp['epoch'] = ex_epoch
    temp['time'] = ex_time
    temp['loss'] = ex_trainloss

    df_temp = pd.DataFrame([temp], index = [time])

    return df_temp

# result of exercise : recall's
def result(state, type, time) :
    
    metrics = re.findall(r'(\S+): (\d+\.\d+)', state)
    temp = {key: float(value) for key, value in metrics}
    temp['type'] = type

    df_temp = pd.DataFrame([temp], index = [time])

    return df_temp

def experiment(list) :
    df = pd.DataFrame()
    i = 0

    while True : 
        #print(list[i])

        # distinguish experiment time and other information
        time, state = list[i].replace('\n','').split(' INFO ')
        time = parse(time)

        """
        Type of states
        1. training time and loss
        2. evluating time and loss
        3. valid result
        4. tet result
        5. Best validation result is updated!
        """
        # 1. training time and loss
        if 'training' in state :
            type = 'training'
            df_temp = etl(state, type, time)

        # 2. evaluating time and loss
        elif 'evaluating' in state :
            type = 'evaluating'
            df_temp = etl(state, type, time)

        # 3. valid result
        elif 'valid result' in state :
            type = 'valid result'
            i = i + 1
            state = list[i]
            df_temp = result(state, type, time)

        # 4. tet result
        elif 'test result' in state :
            type = 'test result'
            i = i + 1
            state = list[i]
            df_temp = result(state, type, time) 

        # 5. Best validation result is updated!
        else :
            temp = {'update' : state.replace('██ ','')}
            df_temp = pd.DataFrame([temp],index = [time])
    
        df = pd.concat([df,df_temp])

        # update i 
        i = i + 1
        if i == len(list) :
            break
        
    return df

# all over data
# parameter information 1
def para_info2(string) :
    Parameters = string.replace('\n','').split('Parameters:')[1].split('=')
    
    rm_word = ['\'','[',']',' ']
    for j in rm_word :
        Parameters[0] = Parameters[0].replace(j,'')
    Parameters[0] = Parameters[0].split(',')

    rm_word2 = [' ','(',')']
    for j in rm_word2 :
        Parameters[1] = Parameters[1].replace(j,'')
    Parameters[1] = Parameters[1].split(',')

    return Parameters

# all over data

# result of over all data : recall's

def AllOverResult(state) :
    
    type = re.findall(r'best (\S+):',state)
    metrics = re.findall(r'(\S+): (\d+\.\d+)', state)
    temp = {key: float(value) for key, value in metrics}
    temp['type'] = type

    return temp

# all over data

# make dataframe showing resul with parameter index 

def AllOver(l) :
    temp = l[2:-1]

    df = pd.DataFrame()

    for i in range(len(temp)) :
        if 'INFO' in temp[i] :
            ind_name, ind = para_info2(temp[i])
            ind = ind[:-1]
            #print(ind)


        else : 
            result = AllOverResult(temp[i])
            df_temp = pd.DataFrame(result, index = [f'{ind}'])
            df = pd.concat([df,df_temp])

        df.index.name = f'{ind_name}'

    return df        

# best result

def AllOverResult2(state) :
    
    type = re.findall(r'([A-Za-z]+):',state)[0]
    metrics = re.findall(r'(\S+): (\d+\.\d+)', state)
    temp = {key: float(value) for key, value in metrics}
    temp['type'] = type

    return temp

# best 

def Best(l) :
    df = pd.DataFrame()
    for i in range(len(l)) :
        if 'INFO' in l[i] :
            best_parameter = para_info(l[i])
        elif ':' in l[i] :
            best_result = AllOverResult2(l[i])
            best_result = pd.DataFrame([best_result])
            temp = best_parameter.join(best_result)
            df = pd.concat([df,temp])

    return df

def para_to_string(para) :
    param = para_info(para)
    str = ''

    for i in param.columns :
        str = str + f', {i} = {param[i][0]}'

    return str