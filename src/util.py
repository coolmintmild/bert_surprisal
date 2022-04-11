import torch
import re
# HuggingFace transformers 패키지 사용
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_model(modelname):
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = AutoModelForMaskedLM.from_pretrained(modelname)
    model.eval()
    model.to('cuda')
    return model, tokenizer

def get_surprisal(word, masked_sentence, model, tokenizer):
    # mi = masked_index
    # 센텐스를 넣고, 토크나이저로 분리해서 각 토큰을 인덱스화 시키고, mask_index가 있는 위치를 지정함.
    tokens = tokenizer.tokenize(masked_sentence)
    for i, token in enumerate(tokens):
        if token.strip() == '[MASK]':
            mi = i
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([indexed_tokens])

    # token_tensor를 cuda 쓰는 형태로 바꿈 (GPU 가동시)
    tokens_tensor = tokens_tensor.to('cuda')
    # 기록을 추적하는 것(과 메모리를 사용하는 것)을 방지하기 위해, 코드 블럭을 with torch.no_grad(): 로 감쌀 수 있다.
    # 특히 변화도(gradient)는 필요없지만, requires_grad=True 가 설정되어 학습 가능한 매개변수를 갖는 모델을 평가(evaluate)할 때 유용하다.
    with torch.no_grad():
        # TypeError: tuple indices must be integers or slices, not tuple 가 발생하여서,
        # model(tokens_tensor) -> model(tokens_tensor)[0]로 수정함
        # model 은 bert_base,tokenizer_base = tp.load_model('bert-base-cased') 로 불러온 모델
        # Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
        predictions = model(tokens_tensor)[0]
        # predictions.shape = torch.Size([1, 34, 28996]) : (batch_size, sequence_length, config.vocab_size)
            # squeeze : 차원감소    div 나누기 (여기서는 1로 나누어서 그대로임)     exp (exponential)  cpu는 cpu로 옮겨서 계산하라는 함수
    # 그 바로 밑 함수(word_weights/word_weights의 sum)까지 보면 softmax 만드는 과정임을 알 수 있음.
    softpred = torch.softmax(predictions[0, mi], 0)
    surprisals = -1 * torch.log2(softpred)
    if len(tokenizer.tokenize(word))==1:                # vocab 에 없는 단어는 [UNK]으로 처리.
        model_word = tokenizer.tokenize(word)[0]
    elif len(tokenizer.tokenize(word))==2:
        if tokenizer.tokenize(word)[0]=='▁':              # word가 2개인데, '_' 'word'로 나뉠시 'word'를 선택
            model_word = tokenizer.tokenize(word)[1]
        elif bool(re.search('[sd.,]', tokenizer.tokenize(word)[1])):              # word가 2개인데, '_word' 's|.|,'로 나뉠시 '_word'를 선택
            model_word = tokenizer.tokenize(word)[0]
        else :
            model_word = '[UNK]'
    else :
        model_word = '[UNK]'
    word_id = tokenizer.convert_tokens_to_ids(model_word)
    word_surprisal = surprisals[word_id]

    return word_surprisal, model_word