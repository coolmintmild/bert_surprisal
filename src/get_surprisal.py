import os
import argparse
import pandas as pd
import util
import matplotlib
matplotlib.use('agg')
# import sys
# sys.path.extend(["C:\\python\\datascience\\lm-diagnostics"])

# test item이 단어별로 line을 이루고 있는데 이를 다시 문장형식으로 바꿔주는 함수.
# <eos>로 문장 단위를 구분
def sentences(words):
    FINAL_TOKENS = {"<eos>", "</S>", "</s>"}
    def gen():
        sentence = []
        for word in words:
            if word in FINAL_TOKENS:
                # 모든 final tokens를 [SEP]로 변경 (BERT 용)
                word = '[SEP]'
                sentence.append(word)
                yield tuple(sentence)
                sentence.clear()
            else:
                sentence.append(word)
    return map(" ".join, gen())


# surprisal을 구하는 함수
def get_surprisals(sentence, model, tokenizer):
    # 단어 단위로 분리 -> 각 단어별 surprisal을 구하기 위해
    tokenized_text = sentence.split(" ")

    surprisals = []
    model_words = []
    for word in tokenized_text:     # 각 단어를 루프
        # if word == '[SEP]':     # 종료라면 surprisals를 계산할 필요가 없으므로, 0으로 두고.
        #     surprisal = 0
        # else :
        # 각 단어를 [MASK]로 변환한 masked sentence를 만듦. The surgeon pricked himself -> The [MASK] pricked himself.
        masked_sentence = '[CLS] ' + sentence.replace(word, '[MASK]')            # 문장 시작을 알리기 위한 [CLS]를 추가함
        surprisal, model_word = util.get_surprisal(word, masked_sentence, model, tokenizer)
        surprisals.append(float(surprisal))
        model_words.append(model_word)

    return surprisals, model_words


# model을 실행시키는 함수
def run_models(args):
    # 아이템 파일을 읽어서 문장단위로 변환
    root_folder = args.stimdir      # args.stimdir에서 지정된 폴더의 경로를 root_folder로 삼음
    result_path = args.resultdir
    folder_list = os.listdir(root_folder)       # root_folder에 있는 모든 폴더를 읽음.

    for folder in folder_list :
        if '.txt' not in folder:        # configuration 만들때 발생되는 오류로 인해 삽입한 코드 (확장자가 .txt인 파일을 부르지 않기 위해)
            print (folder)
            path = f'{root_folder}/{folder}'                # datasets의 폴더 구조 살펴보기
            conditions_df = pd.read_csv(
                os.path.join(path, "items-BERT.tsv"),       # items-BERT.tsv 가 test set 파일 이름-> 본인의 파일이름이 다르다면 수정
                sep="\t", index_col=0
            ).reset_index(drop=True)
            # 단어 단위로 되어있는 데이터를 문장단위로 변경 (items-BERT.tsv을 보면 단어별로 나눠져 있음. 이것을 다시 문장으로 바꾸는 작업)
            sentence_list = [sentence for sentence in sentences(conditions_df['word'])]

            surprisals_list = []
            model_words_list = []
            # 각 문장의 surprisal 계산
            for sentence in sentence_list:
                surprisals, model_words = get_surprisals(sentence, model, tokenizer)
                surprisals_list = surprisals_list + surprisals
                model_words_list = model_words_list + model_words

            conditions_df['surprisal'] = surprisals_list
            conditions_df['model_word'] = model_words_list

            conditions_df.to_excel(f'{result_path}/results_[{args.model}].xlsx', sheet_name='Sheet1')
            # .xlsx 파일로 결과 출력, tsv 및 csv 원할 경우 아래와 같이
            # conditions_df.to_csv(f'{result_path}/results_[{args.model}].tsv', sep='\t', sheet_name='Sheet1')
            print ("EXPORTED RESULTS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimdir", default=None, type=str)        # test set이 있는 directory의 경로
    parser.add_argument("--model", default=None, type=str)           # 사용할 모델의 이름 (HuggingFace) e.g. bert-base-uncased
    parser.add_argument("--resultdir", default=None, type=str)       # result 파일 저장 경로
    args = parser.parse_args()

    print('LOADING MODELS')
    model, tokenizer = util.load_model(args.model)         # 모델의 폴더를 쓰거나(?) 'bert-base-cased'를 쓰거나(?)

    # LM configuration 출력하기
    config_log = open(f'{args.resultdir}/{args.model}_config.txt', 'w')
    config_log.write(str(model.config))
    config_log.close()

    # model = (str(args.bertmodel),bert_model,bert_tokenizer)

    print('RUNNING EXPERIMENTS')
    run_models(args)

# BERT : bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking
# ALBERT : albert-base-v2, albert-large-v2, albert-xlarge-v2, albert-xxlarge-v2
'''
Usage Example

python get_surprisal.py \
--stimdir C:/python/datascience/lm-diagnostics/datasets/surprisal2\ \
--model albert-base-v2 \
--resultdir C:/python/datascience/lm-diagnostics/results
'''
