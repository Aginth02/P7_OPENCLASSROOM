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
import re 

# Importation des jeux de données
dataset_train = pd.read_csv('3_dashbord/data/df_train_final.csv',index_col=0)
dataset_test = pd.read_csv('3_dashbord/data/df_test_final.csv',index_col=0)

target_train = pd.read_csv('3_dashbord/data/target_train_final.csv',index_col=0)
model = pickle.load(open('3_dashbord/data/best_model.pickle', 'rb'))
scaler = pickle.load(open('3_dashbord/data/scaler.pickle','rb'))

dataset_train_nostandar = scaler.inverse_transform(dataset_train)
dataset_train_nostandar = pd.DataFrame(dataset_train_nostandar,columns=dataset_train.columns)

dataset_test_nostandar = scaler.inverse_transform(dataset_test)
dataset_test_nostandar = pd.DataFrame(dataset_test_nostandar,columns=dataset_test.columns,index=dataset_test.index)


id_client = list(dataset_test.index)

#_____________________________________________________________________________________
#____________________________ FONCTION ANALYSE CLIENT_________________________________
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

def distribution_age(X,df_train):
    nb_age = np.round(X['DAYS_BIRTH']/-365.25)
    age_train = np.round(df_train['DAYS_BIRTH']/-365.25)

    fig = sns.displot(age_train)
    fig.refline(x = nb_age,color = "red",lw = 3)
    st.pyplot(fig)
    

    return 

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

def family (X):
    nb_child = int(np.round(X['CNT_CHILDREN']))
    nb_fam_memb = int(np.round(X['CNT_FAM_MEMBERS']))
    
    ls_fam_status=[] ; ls_occupation_type=[]
    
    for i in X.index :
        if 'NAME_FAMILY_STATUS' in i :
            ls_fam_status.append(i)
        if  'OCCUPATION_TYPE' in i :
            ls_occupation_type.append(i)
            
    new_X_fam=X[ls_fam_status] ; new_X_occup=X[ls_occupation_type]
    
    status = new_X_fam[new_X_fam==1].index[0]
    status = status.replace('NAME_FAMILY_STATUS_',"")
    
    if len(new_X_occup[new_X_occup==1]) > 0:
        occupation = new_X_occup[new_X_occup==1].index[0]
        occupation = occupation.replace('OCCUPATION_TYPE_',"")
    else :
        occupation = 'no information'
        
        
    child = "Nombre d'enfant : "+ str(nb_child)
    fam_memb = "Nombre de membre de la famille : " + str(nb_fam_memb)
    status = "Status Familiale : "+ status
    occupation = 'Emploi : '+ occupation
    return child,fam_memb,status,occupation

def distribution_salary(X,df_train):
    salary = np.round(X['AMT_INCOME_TOTAL'],3)
    salary_train = np.round(df_train['AMT_INCOME_TOTAL'],3)

    fig = sns.displot(salary_train)
    fig.refline(x = salary,color = "red",lw = 3)
    st.pyplot(fig)
    return

def credit_salary(X) :
    credit=np.round(X['AMT_CREDIT'],3)
    annuity = np.round(X['AMT_ANNUITY'],3)
    salary = np.round(X['AMT_INCOME_TOTAL'],3)
    duree = np.rint(credit/annuity)
    payment_rate = np.round(X['PAYMENT_RATE'],3)
    annuity_perc = np.round(X['ANNUITY_INCOME_PERC'],3)
    cred_perc = np.round(X['INCOME_CREDIT_PERC'],3)
    
    ls_contrat = []
    for i in X.index :
        if (re.search('^NAME_CONTRACT_TYPE',i)) != None :
            ls_contrat.append(i)
            
    new_X=X[ls_contrat]
    type_contrat = new_X[new_X==1].index[0]
    type_contrat = type_contrat.replace('NAME_CONTRACT_TYPE_',"")
    
    credit= 'Montant du crédit : '+str(credit)+'$'
    annuity= 'Annuités de prêt : '+str(annuity)+'$ par an'
    salary = 'Salaire du client : '+str(salary)+'$ par an'
    payment_rate = 'Taux de paiement : '+str(np.round(payment_rate*100))+'%'
    annuity_perc = "Annuité = "+str(np.round(annuity_perc*100))+'% du Salaire Annuel'
    cred_perc = "Salaire Annuel= "+str(np.round(cred_perc*100))+'% du Crédit '
    type_contrat = 'Type de contrat : '+type_contrat
    duree = 'Durée du prêt : '+str(duree)+' ans'
    return credit,annuity,salary,payment_rate,annuity_perc,cred_perc,type_contrat,duree

def previous_credit(X):
    mean_annuity_homeC= np.round(X['PREV_AMT_ANNUITY_MEAN'],3)
    mean_credit_homeC = np.round(X['PREV_AMT_CREDIT_MEAN'],3)
    mean_time_homeC = np.round(X['PREV_CNT_PAYMENT_MEAN'],3)
    
    mean_credit_BurC = np.round(X['BURO_AMT_CREDIT_SUM_MEAN'],3)
    mean_dette_BurC = np.round(X['BURO_AMT_CREDIT_SUM_DEBT_MEAN'],3)
    
    
    mean_annuity_homeC='Annuité Moyenne : '+str(mean_annuity_homeC)+'$'
    mean_credit_homeC='Crédit Moyen : '+str(mean_credit_homeC)+'$'
    mean_time_homeC=' Durée moyen de remboursement : '+str(mean_time_homeC)+' ans'
    
    mean_credit_BurC="Crédit Moyen :"+str(mean_credit_BurC)+'$'
    mean_dette_BurC="Dettes Moyennes : "+str(mean_dette_BurC)+'$'
    
    return mean_annuity_homeC,mean_credit_homeC,mean_time_homeC,mean_credit_BurC,mean_dette_BurC

def client_similaire(df_test,df_test_nnstandard,name):
    id_client = name
    df=df_test.copy()
    tmp_df = df.sub(df.loc[id_client], axis='columns')
    tmp_series = tmp_df.apply(np.square).apply(np.sum, axis=1)
    tmp_series = tmp_series.sort_values(ascending=True)
    voisins = df_test_nnstandard.loc[tmp_series.index.tolist()[1:6]]
    return voisins.index.tolist()

def info_comp(df_comp,X_comp):
    df_comp['DAYS_BIRTH'] = np.round(df_comp['DAYS_BIRTH']/-365.25)
    X_comp['DAYS_BIRTH'] = np.round(X_comp['DAYS_BIRTH']/-365.25)
    fig, axes = plt.subplots(5, 2,figsize=(18, 40))

    for i in range(len(ls_comp)):
        var=ls_comp[i]
        sns.boxplot(ax=axes[i][0],data=df_comp[var], orient="v")
        axes[i][0].axhline(X_comp[var], color="red", linestyle='dashed')
        axes[i][0].set(title=var, xlabel='', ylabel='')

        sns.histplot(ax=axes[i][1],data=df_comp[var])
        axes[i][1].axvline(X_comp[var], color="red", linestyle='dashed')
        axes[i][1].set(title=var, xlabel='', ylabel='')

    st.pyplot(fig)
        
    return
#_____________________________________________________________________________________
#____________________________ FONCTION ANALYSE GENERALE_______________________________
def graph_analyse_gen(df_train,chiffre):
    
    gender = df_train['CODE_GENDER'].copy()
    age = np.round(df_train['DAYS_BIRTH']/-365.25)
    salary = np.round(df_train['AMT_INCOME_TOTAL'],3)
    credit=np.round(df_train['AMT_CREDIT'],3) ; annuity = np.round(df_train['AMT_ANNUITY'],3)
    duree = np.rint(credit/annuity)
    nb_fam_memb = np.round(df_train['CNT_FAM_MEMBERS']).copy().astype(int)
    mean_credit_homeC = np.round(df_train['PREV_AMT_CREDIT_MEAN'],3)
    mean_credit_BurC = np.round(df_train['BURO_AMT_CREDIT_SUM_MEAN'],3)
    
    dico_graph={1:gender,
               2:age,
               3:nb_fam_memb,
               4:salary,
               5:credit,
               6:duree,
               7:mean_credit_homeC,
               8:mean_credit_BurC,}
    
    dico_legend={1:'Repartition Homme/Femme',
               2:'Age des client',
               3:'Nombre de membre de famille',
               4:'Salaire annuel',
               5:'Montant des crédits',
               6:'Durée des crédits',
               7:'Crédits immobiliers ',
               8:'Crédits chez Bureau Crédit',}
    
    if chiffre == 1 : 
        gender[gender==0]='Homme'
        gender[gender==1]='Femme'
        fig = sns.displot(gender)
        plt.title(dico_legend[1])
    else :
        fig = sns.displot(dico_graph[chiffre])
        plt.title(dico_legend[chiffre])
    
    st.pyplot(fig)
    return


#_____________________________________________________________________________________
#____________________________ CONFIGURATION APPLICATION ______________________________

st.set_page_config('Accord Crédit')
original_title = '<p style="font-family:Arial; color:Red; font-size: 60px;text-align: center;"> Application de prediction de prêt bancaire</p>'
st.markdown(original_title, unsafe_allow_html=True)
st.markdown("***")





selected = option_menu(None, ["Analyse Générale", "Analyse Client"], 
    icons=['Analyse Générale', 'Analyse Client'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
selected


if selected == "Analyse Client":

    form2 = st.form("template_form1")
    analyse_client = form2.selectbox("Choix Clients",id_client,index=0)
    submit_client = form2.form_submit_button("Lancer Analyse")

    if submit_client :        
       
        X = dataset_test.loc[analyse_client]
        X_nostandar = dataset_test_nostandar.loc[analyse_client]

        probability_default_payment = model.predict_proba(pd.DataFrame(X.values,index=X.index).T)[:, 1]
        client = "<h1 style='text-align: center; color: black;'>CLIENT " + str(analyse_client)+"</h1>"
        st.markdown(client, unsafe_allow_html=True)

        with st.expander("Information Personnelles du Client"):

            code_gender=gender(X_nostandar)
            st.write(code_gender)
            age=age(X_nostandar)
            st.write(age)

            distribution_age(X_nostandar,dataset_train_nostandar)





            house = type_house(X_nostandar)
            st.write(house)

            children,fam_member,fam_status,occupation_type=family(X_nostandar)
            st.write(children)
            st.write(fam_member)
            st.write(fam_status)
            st.write(occupation_type)

        with st.expander("Informations sur la demande de Credits du Client"):

            credit,annuity,salary,payment_rate,annuity_perc,cred_perc,type_contrat,duree = credit_salary(X_nostandar)
            st.write(credit)
            st.write(annuity)
            st.write(duree)
            st.write(salary)
            distribution_salary(X_nostandar,dataset_train_nostandar)
            st.write(cred_perc)
            st.write(annuity_perc)
            st.write(payment_rate)
            st.write(type_contrat)

        with st.expander("Informations sur les précédents crédits"):

        
            mean_annuity_homeC,mean_credit_homeC,mean_time_homeC,mean_credit_BurC,mean_dette_BurC=previous_credit(X_nostandar)
            st.markdown("<h5 style='color: black;font-size : 20px;'>Crédit Imobilier</h5>", unsafe_allow_html=True)
            st.write(mean_credit_homeC)
            st.write(mean_annuity_homeC)
            st.write(mean_time_homeC)
            st.markdown("<h6 style='color: black;font-size : 20px;'>Credit Bureau</h6>", unsafe_allow_html=True)
            st.write(mean_credit_BurC)
            st.write(mean_dette_BurC)
        

        defaut = "<h7 style='font-family:Arial; color:Black; font-size: 30px;'>Probabilité de défaut de paiement : "+str(np.round(probability_default_payment[0],2)*100)+" % </h7>"

        st.markdown(defaut,unsafe_allow_html=True)
        if probability_default_payment > 0.5 : 
            st.markdown('<h8 style="font-family:Arial; color:Red; font-size: 50px; text-align: center;">CREDIT REFUSE</h8>',unsafe_allow_html=True)
        else :
            st.markdown('<h8 style="font-family:Arial; color:Red; font-size: 50px; text-align: center;">CREDIT ACCORDE</h8>',unsafe_allow_html=True)

        
        with st.expander("Informations Complémentaire"):
            st.write('Les 5 variables  importantes dans notre modèle sont :') 
            st.write('EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, DAYS_BIRTH, PAYMENT_RATE')

            ls_comp=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'PAYMENT_RATE']
            df_comp = dataset_train_nostandar[ls_comp]
            X_comp = X_nostandar.loc[ls_comp]
            info_comp(df_comp,X_comp)
        

        with st.expander("Clients Similaires"):
            voisins = client_similaire(dataset_test,dataset_test_nostandar,X.name)
            ls_voisin = ', '.join(list(map(str, voisins)))
            st.write(ls_voisin)




else:
    dico_analyse={'GENRE':1,'AGE':2,'MEMBRE DE FAMILLE':3,'REVENUE ANNUEL':4,'MONTANT CREDIT':5,'DUREE CREDIT':6,'CREDIT IMMOBILIER MOYEN':7,'CREDIT MOYEN CHEZ CREDIT DE BUREAU':8}
    
    form3 = st.form("template_form2")
    analyse_general = form3.selectbox("Analyse souhaitée",list(dico_analyse.keys()),index=0)
    submit_analyse = form3.form_submit_button("Lancer Analyse")
    
    if submit_analyse :
        chiffre_analyse=dico_analyse[analyse_general]
        graph_analyse_gen(dataset_train_nostandar,chiffre_analyse)


		
