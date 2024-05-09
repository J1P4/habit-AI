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

class food_recommendation():
  PATH_NAME = 'C:/Users/woobi/Documents/habit/habit-AI/data/'
  PATH_NAME2 = 'C:/Users/woobi/Documents/habit/habit-AI/'
  # PATH_NAME = 'C:/Users/gkstk/OneDrive/Desktop/SangMin/Github/AI/data/'
  # PATH_NAME2 = 'C:/Users/gkstk/OneDrive/Desktop/SangMin/Github/AI/'
    
  # 서버 실행시 한번만 로드 할 수 있도록 할 것
  model = KeyedVectors.load(PATH_NAME + "한국어_음식모델_한상민_v2.kv", mmap='r')
    
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

  def process_food_log(curr_log):
    curr_log['predicted_categories_number'] = 0
    curr_log['predicted_categories_words'] = ""
    curr_log['max_cosim_score'] = 0
    curr_log['most_sim_food'] = ""
    curr_log['reciprocal_rank'] = 0.0
    curr_log['sym_reciprocal_rank'] = 0.0

    for i in range(curr_log.shape[0]):
      food_name = curr_log.loc[i, 'Food Name']
      pre_embedding = reduce_with_food_words(food_name)

      word_embed = np.zeros(shape = (1, len(model["불고기"])))
      if len(pre_embedding) > 0:

        num_words = 0
        for word in pre_embedding:
          word = word.lower()

          if word in model:
            num_words += 1
            word_embed += model[word]

        if num_words != 0:
          word_embed /= num_words

      similarities = cosine_similarity(word_embed, wweia_embeddings)
      to_keep_args = np.argsort(similarities, axis=1)
      indices = np.flip(to_keep_args, axis = 1)

      most_sim_food_row = wweia_data.iloc[indices[0,0], :]
      highest_cat_num = most_sim_food_row['NO']
      highest_cat_words = wweia_food_categories.loc[wweia_food_categories['NO'] == highest_cat_num, '식품명']
      curr_log.loc[i, 'predicted_categories_number'] = highest_cat_num
      curr_log.loc[i, 'predicted_categories_words'] = highest_cat_words.to_list()[0]

    return curr_log

  # Main method
  def run_food_recommandation(input_food_list) :

    input_list= ["wweia_food_category_code", "Food Name", "wweia_food_category_description"]
    curr_log = pd.DataFrame(input_food_list, columns=input_list)

    curr_log = process_food_log(curr_log)

    print("true 출력 ")
    first_list = list(set(curr_log.loc[:,'wweia_food_category_code'].tolist()))
    print(first_list)

    print("pred 출력 ")
    second_list = list(set(curr_log.loc[:,'predicted_categories_number'].tolist()))
    print(second_list)
    
    print("출력 ")
    last_list = first_list + second_list
    last_list = list(set(last_list))
    print(last_list)
    
    category_info_list = []
    for category_num in last_list:
        category_row = self.wweia_food_categories[self.wweia_food_categories['NO'] == category_num].iloc[0]
        category_dict = {
            'foodId': category_row['NO'],
            'name': category_row['식품명'],
            'category': category_row['식품상세분류']
        }
        print(category_dict)
        category_info_list.append(category_dict)
    
    print(category_info_list)

