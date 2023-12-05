import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


#import data
s = pd.read_csv('social_media_usage.csv')

#creating clean function

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

def female_clean(x):
    x = np.where(x == 2, 1, 0)
    return x

#creating subset dataframe
ss = pd.DataFrame({
    'sm_li': s['web1h'].apply(clean_sm),
    'income': np.where(s['income']>9, np.NaN, s['income']),
    'education': np.where(s['educ2']>8, np.NaN, s['educ2']),
    'parent' : s['par'].apply(clean_sm),
    'married': s['marital'].apply(clean_sm),
    'female' : s['gender'].apply(female_clean),
    'age': np.where(s['age']>98, np.NaN, s['age'])
})
ss = ss.dropna(how='any')

#target vector (y)
y = ss[['sm_li']]
x = ss.drop(['sm_li'], axis=1)

#test train split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=987)

# instantiate the model (using the default parameters)
model = LogisticRegression(class_weight="balanced")

# fit the model with data
model.fit(x_train, y_train.values.ravel())


st.markdown("Hello and Welcome!")

st.markdown("Are you a LinkedIn user?")

income_list = list["1 = Less Than $10,000", "2":"10 to under $20,000", "3":"20 to under $30,000","4":"30 to under $40,000","5":"40 to under $50,000","6":"50 to under $75,000", "7":"75 to under $100,000","8":"100 to under $150,000","9":"$150,000 or more","10":"NA"]
education_list = list["1":"Less than High School 1-8", "2":"No Diploma", "3":"High School Graduate", "4":"Some College, No Degree", "5":"Associate Degree","6":"Bachelor Degree","7":"Some postgraduate no degree","8":"Postgraduate or Professional Degree","9":"NA"]
parent_list= list["0":"False", "1":"True"]
married_list= list["0":"False", "1":"True"]
gender_list= list["0":"Male", "1":"Female", "0"-"Other"]

user_income = st.selectbox(label = "What is your income?, Options = income_list")

user_education = st.selectbox(label = "What is your education level?, Options = education_list")

user_parent = st.selectbox(label = "Are you a parent?, Options = parent_list")

user_married = st.selectbox(label = "Are you married?, Options = married_list")

user_female = st.selectbox(label = "What is your gender?, Options = gender_list")

user_age = st.text_input("What is your Age?")


#add button (if true st.button = ) if button ==True: 
#user_complete = st.button("Calculate"):

#creating user data
User = pd.DataFrame(np.array([{user_income}, {user_education},{user_parent},{user_married},{user_female},{user_age}]).reshape(1,-1), 
                    columns=['income','education','parent','married','female','age'])
#Predicting outcome
model.predict(User)

print(f"Predicted class: {model.predict(User)}")

#generating Probability of postivie class (=1)
probs = model.predict_proba(User)

print(f"Probability that this person has a linked in account: {probs[0][1]}")


