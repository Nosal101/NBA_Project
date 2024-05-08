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

#corr_map(data)

#########USUNIĘCIE WIERSZÓW Z KORELACJĄ PONIŻEJ 0.15##########
def remove_rows(df, column_name):
    df.drop(columns=[column_name], inplace=True)
remove_rows(data, 'FG%')
remove_rows(data, 'FT%')
remove_rows(data, '3P%')


#########PODZIAŁ NA SEZONY##########
data_2023 = data[data['Sezon'] == 2023]
data = data[data['Sezon'] != 2023]

###########WYRZUCENIE KOLUMNY SEZON##########
data_2023.drop(columns=['Sezon'], inplace=True)
data.drop(columns=['Sezon'], inplace=True)

###########PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY##########
X_train = data.iloc[:,1:-1]
y_train = data.iloc[:,-1]
X_test = data_2023.iloc[:,1:-1]
y_test = data_2023.iloc[:,-1]

#########################################################
########## TESTOWANIE MODELU DLA DRUŻYNY  ROKU ##########
#########################################################

##########MODELE##########
models = [
    LogisticRegression(random_state=0),
    KNeighborsClassifier(n_neighbors=8,weights = "uniform"),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(n_estimators=100, random_state=0),
    GradientBoostingClassifier(random_state=0, n_estimators=100,learning_rate = 0.1),
    AdaBoostClassifier(random_state=0),
    SVC(kernel='linear'),
    GaussianNB(),
    MLPClassifier(random_state=0, max_iter=1000)
]

##########PREDYKCJA##########
def evaluate_model(model, X_train, y_train, X_test, data_2023):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    selected_indices = np.where((y_pred == 1) | (y_pred == 2))[0]
    selected_players = data_2023.iloc[selected_indices]['PLAYER']
    #print("Zawodnicy dopasowani do kategorii 1, 2 lub 3:")
    #print(selected_players)
    return selected_players

##########LICZBA WYSTĄPIEŃ DLA KAŻDEGO ZAWODNIKA##########
player_counts = {}

def player_count(data_2023, X_train, y_train, X_test, player_counts):
    for model in models:
        selected_players = evaluate_model(model, X_train, y_train, X_test, data_2023)
        for player in selected_players:
            if player in player_counts:
                player_counts[player] += 1
            else:
                player_counts[player] = 1
    #print("\nLiczba wystąpień dla każdego zawodnika:")
    #for player, count in player_counts.items():
        #print(f"{player}: {count}")

player_count(data_2023, X_train, y_train, X_test, player_counts)

##########PODZIAŁ NA ZESPOŁY##########
team1 =[]
team2 =[]

def add_to_group(player_counts):
    sorted_players = sorted(player_counts.items(), key=lambda x: x[1], reverse=True)
    team1.append(sorted_players[0])
    team1.append(sorted_players[1])
    team1.append(sorted_players[2])
    team1.append(sorted_players[3])
    team1.append(sorted_players[4])
    team2.append(sorted_players[5])
    team2.append(sorted_players[6])
    team2.append(sorted_players[7])
    team2.append(sorted_players[8])
    team2.append(sorted_players[9])

    
add_to_group(player_counts)

print('Zespół 1')  #4 dobrze, 1 do drużyny 2
print(team1)
print('Zespół 2')  #2 dobrze, 1 do drużyny 1, 2 źle
print(team2)

#########################################
########## WIZUALIZACJA DANYCH ##########
#########################################
plt.figure(figsize=(12, 6))

# Zespół 1
plt.subplot(1, 2, 1)
plt.title('Zespół 1')
plt.gca().axes.get_yaxis().set_visible(False) # Ukrycie osi Y
for i, (player, _) in enumerate(team1):
    if player == team1[0][0]:
        plt.text(0.5, 0.95 - i*0.1, player, ha='center', color='red')
    else:
        plt.text(0.5, 0.95 - i*0.1, player, ha='center')

# Zespół 2
plt.subplot(1, 2, 2)
plt.title('Zespół 2')
plt.gca().axes.get_yaxis().set_visible(False) # Ukrycie osi Y
for i, (player, _) in enumerate(team2):
    plt.text(0.5, 0.95 - i*0.1, player, ha='center')

plt.tight_layout()
plt.show()