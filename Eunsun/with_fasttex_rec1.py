import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
import fasttext.util
import fasttext
import heapq
import json

class Color:
  Red = '\033[31m'
  Green = '\033[32m'
  Yellow = '\033[33m'
  Reset = '\033[0m'
  Cyan = '\033[36m'
  Magenta = '\033[35m'

print(Color.Yellow + 'fasttex 로드 중(1분 정도 걸림..)' + Color.Reset)
ft = fasttext.load_model('data/cc.ko.300.bin')
print(Color.Yellow + 'fasttext 로드 완료' + Color.Reset)

print(Color.Yellow + 'train 로드 중(1~2분 정도 걸림..)' + Color.Reset)
songs_with_tags = pd.read_json('data/songs_with_tags.json', orient='table')
data = json.load(open('data/processed_fast_train.json'))
train_df = pd.DataFrame(data["data"])
print(Color.Yellow + 'train 로드 완료' + Color.Reset)

sad = pd.read_csv('data/sad_songs.csv')['song id']
sad = sad.values
happy = pd.read_csv('data/happy_songs.csv')['song id']
happy = happy.values
peace = pd.read_csv('data/peace_songs.csv')['song id']
peace = peace.values
light = pd.read_csv('data/light_songs.csv')['song id']
light = light.values
scary = pd.read_csv('data/scary_songs.csv')['song id']
scary = scary.values
comfort = pd.read_csv('data/comfort_songs.csv')['song id']
comfort = comfort.values
cry = pd.read_csv('data/cry_songs.csv')['song id']
cry = cry.values
nervous = pd.read_csv('data/nervous_songs.csv')['song id']
nervous = nervous.values
emo = []
prob = []
sa_active = 0
h_active = 0
p_active = 0
l_active = 0
sc_active = 0
co_active = 0
cr_active = 0
n_active = 0
top_n = 5
max_song_list = 100
found_list = []
found = 0
flag = 1
print(Color.Yellow + '데이터 로드 완료' + Color.Reset)
print()
print(Color.Cyan + "글과 어울리는 태그를 입력해주세요!" + Color.Reset)
print(Color.Cyan + 'ex) 힐링, 쉼, 휴식, 고요'+ Color.Reset)
#user_tag = input()#.split(',')
user_tag='힐링, 쉼, 휴식, 고요'
print('입력한 태그들', user_tag)

print(Color.Cyan + "아래 감정들 중 해당되는 감정과 확률값을 입력해주세요!" + Color.Reset)
print(Color.Cyan + "경쾌한, 고요한, 긴장되는, 잔잔한, 우울한, 울고싶은, 무서운, 편안한, 행복한" + Color.Reset)
print(Color.Cyan + 'ex) 잔잔한 0.7, 편안한 0.3'+ Color.Reset)
#emotions = input().split(',')
emotions = '잔잔한 0.7, 편안한 0.3'.split(',')
print(emotions)

# 감정과 확률 분리
for x in emotions:
    emo.append(x.split()[0])
    prob.append(float(x.split()[1]))
print(emo)
print(prob)

for emo_idx in range(len(emo)):
    emo = emo[emo_idx]
    if emo == '잔잔한':
        p_active = prob[emo_idx]
    if emo == '편안한':
        co_active = prob[emo_idx]
    if emo == '우울한':
        sa_active = prob[emo_idx]
    if emo == '행복한':
        h_active = prob[emo_idx]
    if emo == '경쾌한':
        l_active = prob[emo_idx]
    if emo == '무서운':
        sc_active = prob[emo_idx]
    if emo == '울고싶은':
        cr_active = prob[emo_idx]
    if emo == '긴장되는':
        n_active = prob[emo_idx]

print(Color.Yellow + '태그 유사도 측정 중(시간 오래 걸릴 수 있다.)' + Color.Reset)
user_tag_vectors = ft.get_sentence_vector(user_tag).reshape((1, -1))
print(Color.Yellow + '태그 유사도 측정 완료, train과 유사도 비교 중' + Color.Reset)

rec_result = []

train_df['cos_sim'] = train_df['tag_vec_ft'].apply(lambda _df: cosine_similarity(_df, user_tag_vectors))
## 시간 오래걸리는데 train 사이즈를 줄일까?
print(Color.Yellow + 'train과 유사도 비교 완료' + Color.Reset)

fr = train_df['cos_sim']
n_result = heapq.nlargest(max_song_list, range(len(fr)), key=fr.__getitem__)

#print('active 값:', sa_active, h_active, p_active, l_active, sc_active, co_active, cr_active, n_active)

for i in range(max_song_list):
    sc_list = []
    rec_result = []
    rec_result.append((train_df.iloc[n_result[i]]['곡 리스트'],
                       train_df.iloc[n_result[i]]['cos_sim']))

    user_rec_songs = rec_result[0][0].replace('[', '').replace(']', '').replace("'", '').replace(" ", '').split(
        ',')  # [:100]
    user_rec_songs = list(map(int, user_rec_songs))
    print('similarity', rec_result[0][1])

    for song in user_rec_songs:
        sc = 0
        if sa_active:
            if song in sad:
                sc += sa_active
        if h_active:
            if song in happy:
                sc += h_active
        if p_active:
            if song in peace:
                sc += p_active
        if l_active:
            if song in light:
                sc += l_active
        if sc_active:
            if song in scary:
                sc += sc_active
        if co_active:
            if song in comfort:
                sc += co_active
        if cr_active:
            if song in cry:
                sc += cr_active
        if n_active:
            if song in nervous:
                sc += n_active
        if sc > 0:
            print('song:', song, 'sc:', sc)
            sc_list.append(sc)

            found += 1
            found_list.append(song)

    if found >= top_n:
        break

# sc 높은 순으로 추천
song_result = heapq.nlargest(top_n, range(len(sc_list)), key=sc_list.__getitem__)
fin_song_result = [found_list[song_result[i]] for i in range(top_n)]

print(Color.Red + '!-------------!추천!-------------!' + Color.Reset)
i=1
for idx in fin_song_result:
    print(Color.Red + str(i)+'번째 추천--------!'+ Color.Reset)
    print('곡 제목: ', songs_with_tags.loc[idx, ['곡 제목']].values[0])
    print('아티스트: ', songs_with_tags.loc[idx, ['아티스트 리스트']].values[0][0])
    print('노래 관련 대표 태그: ', songs_with_tags.loc[idx, ['tag max']].values[0])
    print('노래에 해당되는 대표 감정: ', songs_with_tags.loc[idx, ['emotion max']].values[0])
    i+=1
    print()

print(Color.Red + '추천 완료!' + Color.Reset)