#PLAYER - Imie zawodnika
#TEAM - Drużyna 
#AGE - Wiek
#GP - Ilość rozegranych meczów
#W - Ilość wygranych meczów
#L - Ilość przegranych meczów
#MIN - Ilość minut spędzonych na boisku na mecz
#PTS - Ilość zdobytych punktów na mecz
#FGM - Ilość celnych rzutów z gry na mecz
#FGA - Ilość rzutów z gry na mecz
#FG% - Skuteczność rzutów z gry
#3PM - Ilość celnych rzutów za 3 punkty na mecz
#3PA - Ilość rzutów za 3 punkty na mecz
#3P% - Skuteczność rzutów za 3 punkty
#FTM - Ilość celnych rzutów osobistych na mecz
#FTA - Ilość rzutów osobistych na mecz
#FT% - Skuteczność rzutów osobistych
#OREB - Ilość zbiórek ofensywnych na mecz
#DREB - Ilość zbiórek defensywnych na mecz	
#REB - Ilość zbiórek na mecz	
#AST - Ilość asyst na mecz	
#TOV - Ilość strat na mecz	
#STL - Ilość przechwytów na mecz	
#BLK - Ilość bloków na mecz	
#PF	- Ilość fauli na mecz
#FP	- Fantasy Points 
#DD2 - Ilość double-double na sezon	
#TD3 - Ilość triple-double na sezon
#+/- - Plus/Minus
#All Star - Czy zawodnik został wybrany do All Star Game
#Sezon - Sezon w którym zawodnik debiutował w NBA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


nba = 'połączony_rookie_nba.csv'
data = pd.read_csv(nba)


#########USUNIĘCIE WIERSZÓW Z NAZWA DRUŻYNY##########
def remove_rows_with_teams(df):
    df.drop(columns=['TEAM'], inplace=True)
remove_rows_with_teams(data)

#########USUNIĘCIE WIERSZÓW Z 0 w GS##########
def remove_rows_with_zero_gs(df):
    df.drop(df[df['GP'] == 0].index, inplace=True)
remove_rows_with_zero_gs(data)

#########USUNIĘCIE WIERSZÓW Z wygranymi meczami##########
def remove_rows_with_wins(df):
    df.drop(columns=['W'], inplace=True)
remove_rows_with_wins(data)

#########USUNIĘCIE WIERSZÓW Z przegranymi meczami##########
def remove_rows_with_loses(df):
    df.drop(columns=['L'], inplace=True)
remove_rows_with_loses(data)

#########USUNIĘCIE WIERSZÓW Z +/-##########
def remove_rows_with_loses(df):
    df.drop(columns=['+/-'], inplace=True)
remove_rows_with_loses(data)

#########USUNIĘCIE WIERSZÓW Z AGE##########
def remove_rows_with_age(df):
    df.drop(columns=['AGE'], inplace=True)
remove_rows_with_age(data)

#########STANDARDYZACJA##########
def standardization(df,column_name):
    scaler = StandardScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])

standardization(data,'GP')
standardization(data,'MIN')
standardization(data,'PTS')
standardization(data,'FGM')
standardization(data,'FGA')
standardization(data,'3PM')
standardization(data,'3PA')
standardization(data,'FTM')
standardization(data,'FTA')
standardization(data,'OREB')
standardization(data,'DREB')
standardization(data,'REB')
standardization(data,'AST')
standardization(data,'TOV')
standardization(data,'STL')
standardization(data,'BLK')
standardization(data,'PF')
standardization(data,'FP')
standardization(data,'DD2')
standardization(data,'TD3')

#########STANDARDYZACJA##########
def minmaxScal(df,column_name):
    scaler = MinMaxScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])

minmaxScal(data,'FG%')
minmaxScal(data,'3P%')
minmaxScal(data,'FT%')

#########MAPA KORELACJI##########
def corr_map(df):
    data_for_corr = df.drop(['PLAYER','Sezon'], axis=1)

    corr = data_for_corr.corr()
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Mapa korelacji statystyk')
    plt.show()

    correlation_matrix_spearman = data_for_corr.corr(method='spearman')
    plt.figure(figsize=(20, 15))
    sns.heatmap(correlation_matrix_spearman, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Spearman Correlation Heatmap')
    plt.show()

corr_map(data)


