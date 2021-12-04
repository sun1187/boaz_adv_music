# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import konlpy.tag
import tensorflow as tf
from hanspell import spell_checker
import re
from flask import Flask, render_template, request, redirect
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
import fasttext.util
import fasttext
import heapq
import json

app = Flask(__name__)

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '을'
             '를', '으로', '자', '에', '와', '하다', '요', '다', '.', ',']

p = ['가슴통증', '객혈', '고혈압', '관절 통증']
ca = ['구토', '급성 복통', '기침', '다뇨증', '두근거림']
s = ['목 통증', '허리 통증', '배뇨 이상', '변비']
h = ['소화불량/만성 복통', '실신', '월경이상/월경통']
l = ['유방통', '유방덩이', '질 분비물', '콧물/코막힘']
cr = ['혈변', '호흡곤란','황달']
n = ['붉은색 소변', '설사', '소변찔끔증']

def to_nan(x):
  if(x == '-'):
    x = ''
  elif(x == '아니오'):
    x = ''
  elif(x == '아뇨'):
    x = ''
  elif(x == '몰라요'):
    x = ''
  elif(x == '모릅니다'):
    x = ''
  elif(x == '모름'):
    x = ''
  elif(x == '아뇨'):
    x = ''
  elif(x == '아뇨'):
    x = ''
  elif(x == '없습니다'):
    x = ''
  elif(x == '없어요'):
    x = ''
  elif(x == '없음'):
    x = ''
  elif(x == '.'):
    x = ''
  return x

def disease_to_at(t):
    if t in p:
        t = '잔잔한'
    elif t in ca:
        t = '편안한'
    elif t in s:
        t = '우울한'
    elif t in h:
        t = '행복한'
    elif t in l:
        t = '경쾌한'
    elif t in cr:
        t = '무서운'
    elif t in n:
        t = '울고싶은'
    else:
        t = '긴장되는'
    return t

def erase_stopwords(text):
  temp_x = okt.morphs(text, stem=True)
  temp_x = [word for word in temp_x if not word in stopwords]
  temp_x = re.findall(r'\w+', str(temp_x))
  temp_x = ' '.join(map(str, temp_x))

  return ' '.join(re.findall(r'\w+', str(temp_x)))


#메인 페이지 라우팅
@app.route("/")
#@app.route("/index")
def index():
    return render_template('index.html')
#이거 세개 지우면 된다.

# 데이터 예측 처리
#@app.route("/detail", methods=['GET', 'POST'])
#def detail():
@app.route("/text_song", methods=['GET', 'POST'])
def text_song():
    if request.method == 'GET': #기본적 웹방식
        #return render_template('detail.html')
        return render_template('text_song.html')
    if request.method == 'POST': #4가지 데이터 전송한 경우
        # 파라미터를 전달 받습니다.
        cheif = str(request.form['cheif'])
        user_tag = str(request.form['tags'])

        # 데이터 저장
        test_df = {'Chief complaint': cheif}
        new_data = pd.DataFrame([test_df])
        new_data.to_csv('new_data.csv', encoding='utf-8-sig')

        ###########글의 분위기 분류 시작########
        # Change Nan
        new_data = new_data.fillna('-')
        for i in range(len(new_data.columns)):
            new_data[new_data.columns[i]] = new_data.apply(lambda x : to_nan(x[new_data.columns[i]]) , axis = 1 )

        new_data['All'] = new_data['Chief complaint']

        # Check spelling
        new_data['All'] = spell_checker.check(new_data['All']).as_dict()['checked']

        # Erase stopwords
        new_data['All'] = new_data.apply(lambda x : erase_stopwords(x['All']) , axis = 1 )

        new_tfidf = tfidf.transform(new_data['All'])

        # Predict Model
        top_k_result = tf.math.top_k(load_model.predict_proba(new_tfidf), k=3, sorted=True)

        # Save Results
        first = top_k_result[1][0][0], top_k_result.values.numpy()[0][0]
        second = top_k_result[1][0][1], top_k_result.values.numpy()[0][1]

        first_name = dummies[first[0]]
        first_proba = round(first[1]*100, 2)
        second_name = dummies[second[0]]
        second_proba = round(second[1]*100, 2)

        #first = "예측된 첫번째 주요 질병은 "+ first_name +" ("+str(first_proba)+"%) "+"입니다."
        #second = "    두번째 주요 질병은 "+ second_name +" ("+str(second_proba)+"%) "+"입니다."
        ###########글의 분위기 분류 완료########

        ##########노래 추천 시작#############3
        #print('first_proba!!!!!!!', first_proba)
        #print('second_proba!!!!!!!', second_proba)
        first_proba /= 100
        second_proba /= 100
        rec_result = []
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
        sc_list = []
        found = 0
        flag = 1

        # 값 저장
        emo = []
        prob = []
        first_name = disease_to_at(first_name)
        second_name = disease_to_at(second_name)
        emo.append(first_name)
        emo.append(second_name)
        #print('first_proba!!!!!!!', first_proba)
        #print('second_proba!!!!!!!', second_proba)
        prob.append(first_proba)
        prob.append(second_proba)

        #print('emo', emo)
        #print('prob', prob)
        for emo_idx in range(len(emo)):
            #print(prob[emo_idx])
            emo_item = emo[emo_idx]
            if emo_item == '잔잔한':
                p_active = prob[emo_idx]
            if emo_item == '편안한':
                co_active = prob[emo_idx]
            if emo_item == '우울한':
                sa_active = prob[emo_idx]
            if emo_item == '행복한':
                h_active = prob[emo_idx]
            if emo_item == '경쾌한':
                l_active = prob[emo_idx]
            if emo_item == '무서운':
                sc_active = prob[emo_idx]
            if emo_item == '울고싶은':
                cr_active = prob[emo_idx]
            if emo_item == '긴장되는':
                n_active = prob[emo_idx]
        #print(prob)
        #print('감정!!!!!1', sa_active, h_active, p_active, l_active, sc_active, co_active, cr_active, n_active)
        user_tag_vectors = ft.get_sentence_vector(user_tag).reshape((1, -1))
        train_df['cos_sim'] = train_df['tag_vec_ft'].apply(lambda _df: cosine_similarity(_df, user_tag_vectors))
        fr = train_df['cos_sim']
        n_result = heapq.nlargest(max_song_list, range(len(fr)), key=fr.__getitem__)

        #print(sa_active, h_active, p_active, l_active, sc_active, co_active, cr_active, n_active)
        #노래 score 계산
        for i in range(max_song_list):
            rec_result = []
            rec_result.append((train_df.iloc[n_result[i]]['곡 리스트'],
                               train_df.iloc[n_result[i]]['cos_sim']))

            user_rec_songs = rec_result[0][0].replace('[', '').replace(']', '').replace("'", '').replace(" ", '').split(
                ',')  # [:100]
            user_rec_songs = list(map(int, user_rec_songs))
            #print('similarity', rec_result[0][1])
            #print(user_rec_songs)

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
        fin_song_result = []
        #print(song_result)
        for i in range(top_n):
            #print(i)
            #print(song_result[i])
            #print(found_list[song_result[i]])
            fin_song_result.append(found_list[song_result[i]])
        #fin_song_result = [found_list[song_result[i]] for i in range(top_n)]

        # 추천 완료
        final_song = []
        for idx in fin_song_result:
            final_song.append((songs_with_tags.loc[idx, ['곡 제목']].values[0],
                        songs_with_tags.loc[idx, ['아티스트 리스트']].values[0][0],
                        songs_with_tags.loc[idx, ['tag max']].values[0],
                        songs_with_tags.loc[idx, ['emotion max']].values[0]
                        ))

        first_mu = "글의 주요 분위기 top1: "+ first_name +" ("+str(first_proba*100)+"%) "+"입니다."
        second_mu = "글의 주요 분위기 top2: "+ second_name +" ("+str(second_proba*100)+"%) "+"입니다."


        first_song = "*Top1* 곡이름: "+ final_song[0][0]
        first_artist = "아티스트 이름: "+final_song[0][1]
        first_tag = "관련 태그: "+final_song[0][2]
        first_mood = " 노래의 분위기: "+final_song[0][3]

        second_song = "*Top2* 곡이름: "+ final_song[1][0]
        second_artist = "아티스트 이름: "+final_song[1][1]
        second_tag = "관련 태그: "+final_song[1][2]
        second_mood = " 노래의 분위기: "+final_song[1][3]

        third_song = "*Top3* 곡이름: "+ final_song[2][0]
        third_artist = "아티스트 이름: "+final_song[2][1]
        third_tag = "관련 태그: "+final_song[2][2]
        third_mood = " 노래의 분위기: "+final_song[2][3]

        fourth_song = "*Top4* 곡이름: "+ final_song[3][0]
        fourth_artist = "아티스트 이름: "+final_song[3][1]
        fourth_tag = "관련 태그: "+final_song[3][2]
        fourth_mood = " 노래의 분위기: "+final_song[3][3]

        fifth_song = "*Top5* 곡이름: "+ final_song[4][0]
        fifth_artist = "아티스트 이름: "+final_song[4][1]
        fifth_tag = "관련 태그: "+final_song[4][2]
        fifth_mood = " 노래의 분위기: "+final_song[4][3]

        #return render_template('detail.html',
        return render_template('text_song.html',
                        first_mu = first_mu, second_mu = second_mu,
                        first_song = first_song, first_artist = first_artist, first_tag = first_tag, first_mood = first_mood,
                        second_song = second_song, second_artist = second_artist, second_tag = second_tag, second_mood = second_mood,
                        third_song = third_song, third_artist = third_artist, third_tag = third_tag, third_mood = third_mood,
                        fourth_song = fourth_song, fourth_artist = fourth_artist, fourth_tag = fourth_tag, fourth_mood = fourth_mood,
                        fifth_song = fifth_song, fifth_artist = fifth_artist, fifth_tag = fifth_tag, fifth_mood = fifth_mood
                        )
    else:
        return render_template('text_song.html')
        #return render_template('detail.html')

if __name__ == '__main__':
    # 모델 및 데이터 로드
    with open('model/level2_tfidf_vectorizer.pkl', 'rb') as b:
        tfidf = pickle.load(b)

    with open('model/level2_estimator.pkl', 'rb') as c:
        load_model = pickle.load(c)

    with open('data/level2_dummies.txt', 'rb') as d:
        dummies = pickle.load(d)

    ft = fasttext.load_model('data/cc.ko.300.bin')
    songs_with_tags = pd.read_json('data/songs_with_tags.json', orient='table')
    data = json.load(open('data/processed_fast_train.json'))
    train_df = pd.DataFrame(data["data"])
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

    print('데이터 로드 완료!')

    okt = konlpy.tag.Okt() # 객체 생성
    app.run(debug = True)
