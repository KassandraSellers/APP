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

st.markdown("Answer the questions to find out!")

income_list = list("1-Less Than $10,000", "2-10 to under $20,000", "3-20 to under $30,000","4-30 to under $40,000","5-40 to under $50,000","6-50 to under $75,000", "7-75 to under $100,000","8-100 to under $150,000","9-$150,000 or more")
education_list = list("1-Less than High School 1-8", "2-No Diploma", "3-High School Graduate", "4-Some College, No Degree", "5-Associate Degree","6-Bachelor Degree","7-Some postgraduate no degree","8-Postgraduate or Professional Degree")
parent_list= list("0 -False", "1-True")
married_list= list("0-False", "1-True")
gender_list= list("0-Male", "1-Female", "3-Other")

#Income Variable Assignment
    match income_list:
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


match education_list:
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

match parent_list:
        case "0 -False":
            parent = 0
        case "1-True":
            parent = 1

match married_list:
        case "0 -False":
            married = 0
        case "1-True":
            married = 1

match gender_list:
        case "0-Male":
            gender = 0
        case "1-Female":
            gender = 1
        case "3-Other":
            gender = 0
       
user_income = st.selectbox(label = "What is your income?", options = income_list)

user_education = st.selectbox(label = "What is your education level?", options = education_list)

user_parent = st.selectbox(label = "Are you a parent?", options = parent_list)

user_married = st.selectbox(label = "Are you married?", options = married_list)

user_female = st.selectbox(label = "What is your gender?", options = gender_list)

user_age = st.text_input("What is your Age?")


#add button (if true st.button = ) if button ==True: 
user_complete = st.button("Calculate")

if user_compete == True:

#creating user data
    User = pd.DataFrame(np.array([{income}, {education},{parent},{married},{female},{user_age}]).reshape(1,-1), 
                    columns=['income','education','parent','married','female','age'])
#Predicting outcome
    model.predict(User)

    print(f"Predicted class: {model.predict(User)}")

#generating Probability of postivie class (=1)
    probs = model.predict_proba(User)

    print(f"Probability that this person has a linked in account: {probs[0][1]}")


