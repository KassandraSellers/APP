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

st.markdown("<h3 style='text-align: center; color: grey;'>Hello and Welcome!</h3>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: grey;'>Are you a LinkedIn user?</h4>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: grey;'>Answer the questions to find out!</h4>", unsafe_allow_html=True)

      
user_income = st.selectbox(label = "What is your income?", options = ("1 - Less Than $10,000", "2 - 10 to under $20,000", "3 - 20 to under $30,000","4 - 30 to under $40,000","5 - 40 to under $50,000","6 - 50 to under $75,000", "7 - 75 to under $100,000","8 - 100 to under $150,000","9 - $150,000 or more"))

user_education = st.selectbox(label = "What is your education level?", options = ("1 - Less than High School 1-8", "2 - No Diploma", "3 - High School Graduate", "4 - Some College, No Degree", "5 - Associate Degree","6 - Bachelor Degree","7 - Some postgraduate no degree","8 - Postgraduate or Professional Degree"))

user_parent = st.selectbox(label = "Are you a parent?", options = ("0 - False", "1 - True"))

user_married = st.selectbox(label = "Are you married?", options = ("0 - False", "1 - True"))

user_female = st.selectbox(label = "What is your gender?", options = ("0 - Male", "1 - Female", "3 - Other"))

user_age = st.slider("Choose Age")


match user_income:
    case "1-Less Than $10,000":
        income = 1
    case "2-10 to under $20,000":
        income = 2
    case "3-20 to under $30,000":
        income = 3
    case "4-30 to under $40,000":
        income = 4
    case "5-40 to under $50,000":
        income = 5
    case "6-50 to under $75,000":
        income = 6
    case "7-75 to under $100,000":
        income = 7
    case "8-100 to under $150,000":
        income = 8
    case "9-$150,000 or more":
        income = 9


match user_education:
    case "1-Less than High School 1-8":
        education = 1
    case "2-No Diploma":
        education = 2
    case "3-High School Graduate":
        education = 3
    case "4-Some College, No Degree":
        education = 4
    case "5-Associate Degree":
        education = 5
    case "6-Bachelor Degree":
        education = 6
    case "7-Some postgraduate no degree":
        education = 7
    case "8-Postgraduate or Professional Degree":
        education = 8

match user_parent:
    case "0 -False":
        parent = 0
    case "1-True":
        parent = 1

match user_married:
    case "0-False":
        married = 0
    case "1-True":
        married = 1

match user_female:
    case "0-Male":
        female = 0
    case "1-Female":
        female = 1
    case "3-Other":
        female = 0


#add button (if true st.button = ) if button ==True: 
user_complete = st.button("Calculate")

if user_complete == True:

#creating user data
    User = pd.DataFrame({
    'income': income,
    'education': education,
    'parent' : parent,
    'married': married,
    'female' : female,
    'age': user_age
},index=[0])

    
#Predicting outcome
    model.predict(User)

#generating Probability of postivie class (=1)
    probs = model.predict_proba(User)


    st.write("Predicted class:",model.predict(User))
    st.write("Probability that this person has a linked in account:",round(probs[0][1],2))

