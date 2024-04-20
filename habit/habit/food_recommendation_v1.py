#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import re
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import pickle
import math
import os
class food_recommendation():
  PATH_NAME = os.path.dirname(os.path.abspath(__file__))+ '/'
  # PATH_NAME2 = 'C:/Users/woobi/Documents/habit-AI/'
  # PATH_NAME = 'C:/Users/gkstk/OneDrive/Desktop/SangMin/Github/AI/data/'
  # PATH_NAME2 = 'C:/Users/gkstk/OneDrive/Desktop/SangMin/Github/AI/'
    
  # 서버 실행시 한번만 로드 할 수 있도록 할 것
  model = KeyedVectors.load(PATH_NAME + "한국어_음식모델_한상민.kv", mmap='r')
  
  with open(PATH_NAME + 'wweia_synonym_cats.pickle', 'rb') as handle:
    wweia_synonym_cats = pickle.load(handle)
    
  wweia_food_categories = pd.read_csv(PATH_NAME + 'wweia_food_categories_addtl.csv')
  
  wweia_data = pd.read_csv(PATH_NAME + 'wweia_data.csv')
  
  wweia_embeddings = pd.read_csv(PATH_NAME + 'word_embeddings.csv', delimiter = ",")
  
  stop_words = ['가', '걍', '것', '고', '과', '는', '도', '들', '등', '때', '로', '를', '뿐', '수', '아니', '않', '없', '에', '에게', '와', '으로', '은', '의', '이', '이다', '있', '자', '잘', '좀', '하다', '한', '조각', '개', '것', '대', '소' ,'단계', '등급', '포함', '미니', '개입']
  # 여기까지
  
  def reduce_with_food_words(rough_phrase):
    korean_string = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", " ", rough_phrase)
    okt = Okt()
    token = okt.morphs(korean_string, stem=True)

    return token

  def process_food_log(curr_log, self):
    curr_log['predicted_categories_number'] = 0
    curr_log['predicted_categories_words'] = ""
    curr_log['max_cosim_score'] = 0
    curr_log['most_sim_food'] = ""
    curr_log['reciprocal_rank'] = 0.0
    curr_log['sym_reciprocal_rank'] = 0.0

    for i in range(curr_log.shape[0]):
      food_name = curr_log.loc[i, 'Food Name']
      pre_embedding = self.reduce_with_food_words(food_name)

      word_embed = np.zeros(shape = (1, len(self.model["불고기"])))
      if len(pre_embedding) > 0:

        num_words = 0
        for word in pre_embedding:
          word = word.lower()

          if word in self.model:
            num_words += 1
            word_embed += self.model[word]

        if num_words != 0:
          word_embed /= num_words

      similarities = cosine_similarity(word_embed, self.wweia_embeddings)
      to_keep_args = np.argsort(similarities, axis=1)
      indices = np.flip(to_keep_args, axis = 1)

      most_sim_food_row = self.wweia_data.iloc[indices[0,0], :]
      highest_cat_num = most_sim_food_row['NO']
      highest_cat_words = self.wweia_food_categories.loc[self.wweia_food_categories['NO'] == highest_cat_num, '식품명']
      curr_log.loc[i, 'predicted_categories_number'] = highest_cat_num
      curr_log.loc[i, 'predicted_categories_words'] = highest_cat_words.to_list()[0]

    return curr_log

  # Main method
  def run_food_recommandation(input_food_list,self) :

    input_list= ["wweia_food_category_code", "Food Name", "wweia_food_category_description"]
    curr_log = pd.DataFrame(input_food_list, columns=input_list)

    curr_log = self.process_food_log(curr_log, self)

    print("true 출력 ")
    print(set(curr_log.loc[:,'wweia_food_category_code'].tolist()))

    print("pred 출력 ")
    print(set(curr_log.loc[:,'predicted_categories_number'].tolist()))

