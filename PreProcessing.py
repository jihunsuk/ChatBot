import os
import re

path_dir = "./data/origin_data"
file_list = os.listdir(path_dir)
file_list.sort()

pattern = "[^ ㄱ-ㅣ가-힣?!]+" # 띄워쓰기, 한글, ?, !, .을 제외한 모든 문자 정규식

fw = open('./data/preprocessed_data/Conversation_Data.txt', 'w') # 데이터를 모아놓은 txt

for file_name in file_list:
    try:
        f = open(path_dir+"/"+file_name, "r")
        for line in f:
            regex = re.compile(pattern)
            word = regex.sub("", line)
            word = word.strip()
            if word != "":
                fw.write(word+"\n")
        f.close()

    except:
        print("Error at "+file_name)
        f.close()

fw.close()