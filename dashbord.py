import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
# Model and performance
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
import lightgbm
import pickle

# Importation des jeux de données
dataset_train = pd.read_csv('data/df_train_final.csv',index_col=0)
dataset_test = pd.read_csv('data/df_test_final.csv',index_col=0)

target_train = pd.read_csv('data/target_train_final.csv',index_col=0)
model = pickle.load(open('data/best_model.pickle', 'rb'))
scaler = pickle.load(open('data/scaler.pickle','rb'))

dataset_train_nostandar = scaler.inverse_transform(dataset_train)
dataset_train_nostandar = pd.DataFrame(dataset_train_nostandar,columns=dataset_train.columns)

dataset_test_nostandar = scaler.inverse_transform(dataset_test)
dataset_test_nostandar = pd.DataFrame(dataset_test_nostandar,columns=dataset_test.columns,index=dataset_test.index)


id_client = list(dataset_test.index)

#_____________________________________________________________________________________
#____________________________ FONCTION _______________________________________________
def gender(X):
    if X['CODE_GENDER']== 0 :
        gender = 'Masculin'
    else :
        gender= 'Féminin'
    code_gender='Sexe : '+ gender
    return code_gender

def age(X):
    nb_age = np.round(X['DAYS_BIRTH']/-365.25)
    age = 'Age : '+ str(nb_age)+ ' ANS'
    return age 

def distribution_age(X,df_train,form):
    nb_age = np.round(X['DAYS_BIRTH']/-365.25)
    age_train = np.round(df_train['DAYS_BIRTH']/-365.25)
    fig =plt.figure(figsize=(10,10))
    sns.displot(age_train)
    plt.axvline(nb_age, 0,90,color='red')
    form.pyplot(fig)
    return fig

def type_house(X):
    ls_house=[]
    for i in X.index:
        if 'NAME_HOUSING_TYPE' in i :
            ls_house.append(i)
    new_X=X[ls_house]
    house = new_X[new_X==1].index[0]
    house = house.replace('NAME_HOUSING_TYPE_',"")
    type_house = 'Type de logement : '+ house
    return type_house


#_____________________________________________________________________________________
#____________________________ CONFIGURATION APPLICATION ______________________________

st.set_page_config('Accord Crédit')
original_title = '<p color:Red; font-size: 50px;text-align: center;">Accord prêt bancaire: Analyse détaillée</p>'
st.markdown(original_title, unsafe_allow_html=True)
st.markdown("***")
st.write("""Application de prediction qu'un client de la banque "Prêt à dépenser" ne rembourse pas son prêt.
""")


selected = option_menu(None, ["Analyse Générale", "Analyse Client"], 
    icons=['Analyse Générale', 'Analyse Client'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
selected


if selected == "Analyse Client":

    form2 = st.form("template_form1")
    analyse_client = form2.selectbox("Choix Clients",id_client,index=0)
    submit_client = form2.form_submit_button("Lancer Analyse")

    if submit_client :        
        #form3 = st.form("template_form2")
       
        X = dataset_test.loc[analyse_client]
        X_nostandar = dataset_test_nostandar.loc[analyse_client]

        probability_default_payment = model.predict_proba(pd.DataFrame(X.values,index=X.index).T)[:, 1]
        client = 'client ' + str(analyse_client)
        st.write(client)


        code_gender=gender(X_nostandar)
        st.write(code_gender)
        age=age(X_nostandar)
        st.write(age)

        nb_age = np.round(X_nostandar['DAYS_BIRTH']/-365.25)
        age_train = np.round(dataset_train_nostandar['DAYS_BIRTH']/-365.25)

        fig= sns.displot(age_train)
        fig.refline(x = nb_age,color = "red",lw = 3)
        #fig.axvline(nb_age, 0,90,color='red')
        st.pyplot(fig)

        #distribution_age(X_nostandar,dataset_train_nostandar,form3)





        house = type_house(X_nostandar)
        st.write(house)

        defaut = "Probabilité de défaut de paiement : "+str(probability_default_payment[0])

        st.write(defaut)
        if probability_default_payment > 0.5 : 
            st.write("""CREDIT REFUSE""")
        else :
            st.write("""CREDIT ACCORDE""")

        #submit_fin = form3.form_submit_button("Fin")



else:
    st.write('ok')


		
