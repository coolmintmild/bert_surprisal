# bert_surprisal

## Example
<pre>
python get_surprisal.py --stimdir ../datasets/surprisal --model bert-base-uncased --resultdir ../results
</pre>

## source
* get_surprisal.py: surprisal (결과값) 구하기
* util.py: model 부르기 & surprisal 구하기 위한 함수들
* get_tsv_formatted.py: test set을 get_surprisal.py에서 이용할 수 있는 형식으로 변환하는 파일

## dataset
* surprisal/items-BERT.tsv: get_surprisal.py에서 stimulus로 이용하는 파일-> 이 형식으로 수정해야 get_surprisal의 수행됨
* formatting/NPZ-plausibility.xlsx: original test set 
  * get_tsv_formatted.py (혹은 자신의 script)를 이용하여 get_surprisal.py에서 요구하는 형식(단어단위 line)으로 변환하여 사용
  * 이 파일의 형식에 맞게 본인이 원하는 test set을 구성한 후 자신의 syntactic test 수행 가능
  * 혹은 처음부터 items-BERT.tsv의 형식에 맞게 test set을 구성하여 사용할 수도 있음.

## result
* surprisal column이 각 단어의 surprisal 값 (기댓값/확률)
* target region의 surprsisal 값의 평균을 조건마다 비교하여 model의 performance를 측정


## test set
> a. As the woman edited the magazine about fishing amused all the reporters.\
> b. As the woman sailed the magazine about fishing amused all the reporters.\
> c. As the woman edited, the magazine about fishing amused all the reporters.\
> d. As the woman edited, the magazine about fishing amused all the reporters.

* a & b: garden-path sentences\
* a: plausible conditions "edited the magazine"\
* b: implausible conditions "?sailed the magazine"\\

a조건의 critical(disambiguation) region ("amsued")가 b 조건에 비해 RT/surprisal이 높을 것으로 예상됨.

## surprisal
* surprisal= the log inverse probability of a target word ([mask] token)
* probability= softmax 출력값을 이용


