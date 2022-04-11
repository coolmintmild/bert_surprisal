import pandas as pd
import re
'''
이 파일은 활용 예시가 없습니다. 하지만 다른 파일처럼 이 파일을 실행하면 
여러분이 직접 코드를 분석하여 어떻게 변환을 하였는지 생각해보세요.
기본적으로 NPZ-transitivity.xlsx의 Sentence과 region을 items-BERT.tsv로 바꿔주는 파일입니다.
'''

# 엑셀형식 문장이 있는 파일의 경로를 지정 (엑셀형식은 파일 참조)
file_path = 'C:/python/datascience/lm-diagnostics/datasets/formatting/'     # original test set의 폴더 경로
df = pd.read_excel(file_path+'NPZ-transitivity.xlsx')                       # orginal test set의 파일 경로
colnames = ["sent_index","word_index","word","region"]+list(df.keys()[3:])
newdf = pd.DataFrame(columns=colnames)

model = input("Press 1 for Transformer, 2 for LSTM.")
# BERT는 동사를 2단어로 나누어서 보기 때문에, 별도로 ,.을 나눌 필요가 없지만 LSTM은 나누어 볼 필요가 있을 수 있음.
# 만약에 BERT도 .,를 따로 단어로 취급할 계획이면 BERT용 파일도 LSTM방식으로 포매팅 하면됨.

for idx in range(0, len(df)):
    if model == '1':
        words = df.Sentence[idx].split()+['<eos>']
    else :
        words = re.sub(r'([,.])', r' \1', df.Sentence[idx]).split()+['<eos>']
    wordlen = len(words)
    sent_index = [df.Stimuli[idx]]*wordlen
    temp = pd.DataFrame(sent_index, columns=["sent_index"])
    temp['word_index'] = list(range(0, wordlen))
    temp['word'] = words
    region = df.region[idx].split()
    if (model == '2') & (',' in words):
        region.insert(words.index(','), 'Comma')
    temp['region'] = region + ['Rest']*(wordlen-len(region)-1)+['End']
    for i in range(3, len(df.keys())):
        temp[df.keys()[i]] = df[df.keys()[i]][idx]
    newdf=pd.concat([newdf, temp])

newdf.to_csv(f'{file_path}/items-{model}.tsv', sep="\t", index=None)


# for idx in range(0, len(df)):
#     words = df.Sentence[idx].replace(".", "").split()+['.', '<eos>']
#     wordlen = len(words)
#     sent_index = [df.Stimuli[idx]]*wordlen
#     temp = pd.DataFrame(sent_index, columns=["sent_index"])
#     temp['word_index'] = list(range(0, wordlen))
#     temp['word'] =words
#     temp['region'] = df.Production[idx].split()+['End', 'End']
#     temp['condition'] = [df.Type[idx]]*wordlen
#     newdf=pd.concat([newdf, temp])


