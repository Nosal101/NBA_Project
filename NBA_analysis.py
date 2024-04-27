#Player - Imie zawodnika
#Pos - Pozycja na boisku                                            #One_hot_encoding                                         
#Age = Wiek gracz                                                   #Standardizacja
#Tm - Drużyna                                                       #One_hot_encoding
#G - Rozegrane mecze                                                #Standardizacja
#GS - Rozegrane mecze w pierwszym składzie                          #Standardizacja
#MP - Minuty na mecz                                                #Standardizacja
#FG - Liczba trafionych rzutów z gry na mecz                        #Standardizacja
#FGA - Liczba prób rzutów na mecz                                   #Standardizacja
#FG% - Procent trafionych rzutów                                    #MinMaxScaler
#3P - Liczba trafionych rzutów za 3                                 #Standardizacja
#3PA - Liczba prób za 3 na mecz                                     #Standardizacja
#3P% - Procent trafionych rzutów za 3                               #MinMaxScaler
#2P - Liczba trafionych rzutów za 2                                 #Standardizacja
#2PA - Liczba prób za 2 na mecz                                     #Standardizacja
#2P% - Procent trafionych rzutów za 2                               #MinMaxScaler
#eFG% - Procent efektywnych rzutów                                  #MinMaxScaler
#FT - Liczba trafionych osobistych na mecz                          #Standardizacja
#FTA - Liczba prób z osobistych na mecz                             #Standardizacja
#FT% - Procent trafionych rzutów osobistych                         #MinMaxScaler
#ORB - Ofensywne zbiórki na mecz                                    #Standardizacja
#DRB - Defensywne zbiórki na meczz                                  #Standardizacja
#TRB - Całkowita suma zbiórek na mecz                               #Standardizacja
#AST - Astsy na mecz                                                #Standardizacja
#STL - Przechwyty na mecz                                           #Standardizacja
#BLK - Bloki na mecz                                                #Standardizacja
#TOV - Straty na mecz                                               #Standardizacja
#PF - Faule na mecz                                                 #Standardizacja
#PTS - Punkty na mecz                                               #Standardizacja
#All_stars - Czy zawodnik był All-Starsem
#MVP - Czy zawodnik był MVP
#Sezon - Sezon w którym grał zawodnik

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


nba = 'połączony_nba.csv'
data = pd.read_csv(nba)

########POŁĄCZENIE POZYCJI##########
def combine_position(df):
    df['Pos'] = df['Pos'].str.split('-').str[0]
combine_position(data)

#########ZMIANA POZYCJI##########
def position_correction(df):
    df.loc[df['Pos'].isin(['PG', 'SG']), 'Pos'] = 'G'
    df.loc[df['Pos'].isin(['SF', 'PF']), 'Pos'] = 'F'
position_correction(data)

#########USUNIĘCIE WIERSZÓW Z 0 w GS##########
def remove_rows_with_zero_gs(df):
    df.drop(df[df['GS'] == 0].index, inplace=True)
remove_rows_with_zero_gs(data)

#########USUNIĘCIE WIERSZÓW Z NAZWA DRUŻYNY##########
def remove_rows_with_teams(df):
    df.drop(columns=['Tm'], inplace=True)
remove_rows_with_teams(data)

#########STANDARDYZACJA##########
def standardization(df,column_name):
    scaler = StandardScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])

standardization(data,'Age')
standardization(data,'G')
standardization(data,'GS')
standardization(data,'MP')
standardization(data,'FG')
standardization(data,'FGA')
standardization(data,'3P')
standardization(data,'3PA')
standardization(data,'2P')
standardization(data,'2PA')
standardization(data,'FT')
standardization(data,'FTA')
standardization(data,'ORB')
standardization(data,'DRB')
standardization(data,'TRB')
standardization(data,'AST')
standardization(data,'STL')
standardization(data,'BLK')
standardization(data,'TOV')
standardization(data,'PF')
standardization(data,'PTS')

#########STANDARDYZACJA##########
def minmaxScal(df,column_name):
    scaler = MinMaxScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])

minmaxScal(data,'FG%')
minmaxScal(data,'3P%')
minmaxScal(data,'2P%')
minmaxScal(data,'eFG%')
minmaxScal(data,'FT%')

#########MAPA KORELACJI##########
def corr_map(df):
    data_for_corr = df.drop(['Player','Sezon','Pos'], axis=1)

    corr = data_for_corr.corr()
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Mapa korelacji statystyk')
    plt.show()

    correlation_matrix_spearman = data_for_corr.corr(method='spearman')
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix_spearman, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Spearman Correlation Heatmap')
    plt.show()
#corr_map(data)

#########USUNIĘCIE WIERSZÓW Z KORELACJĄ PONIŻEJ 0.1##########
def remove_rows(df, column_name):
    df.drop(columns=[column_name], inplace=True)
remove_rows(data, 'Age')
remove_rows(data, '3P%')

#########KODOWANIE ZMIENNYCH KATEGORIALNYCH##########
def one_hot_encoding(df,column_name):
    label_encoder = LabelEncoder()
    df[column_name] = label_encoder.fit_transform(df[[column_name]])
one_hot_encoding(data,'Pos')

#########PODZIAŁ NA SEZONY##########
data_2023 = data[data['Sezon'] == 2023]
data = data[data['Sezon'] != 2023]

##########WYKRESY PUDEŁKOWE DLA POZYCJI###########
def box_plot(data):
    features = [col for col in data.columns if col not in ['Pos', 'All_stars']]
    for feature in features:
        plt.figure(figsize=(10, 6))
    
        sns.boxplot(x='Pos', y=feature, data=data[data['Pos'].isin([0, 1, 2])])
        plt.title(f'Distribution of {feature} by Position')
        plt.xlabel('Position')
        plt.ylabel(feature)
        plt.xticks([0, 1, 2], ['Center', 'Guard', 'Forward'])
        plt.show()
#box_plot(data)

##########PODZIAŁ NA POZYCJE##########
data_pos0 = data[data['Pos'] == 0]
data_pos1 = data[data['Pos'] == 1]
data_pos2 = data[data['Pos'] == 2]

data_pos0_2023 = data_2023[data_2023['Pos'] == 0]
data_pos1_2023 = data_2023[data_2023['Pos'] == 1]
data_pos2_2023 = data_2023[data_2023['Pos'] == 2]

##########FILTRACJA DLA POZYCJI##########
filtered_data0 = data_pos0[(data_pos0['All_stars'].isin([0, 1, 2, 3]))]
filtered_data0.drop(columns=['Sezon'], inplace=True)

filtered_data1 = data_pos0[(data_pos0['All_stars'].isin([0, 1, 2, 3]))]
filtered_data1.drop(columns=['Sezon'], inplace=True)

filtered_data2 = data_pos0[(data_pos0['All_stars'].isin([0, 1, 2, 3]))]
filtered_data2.drop(columns=['Sezon'], inplace=True)

##########WYKRESY ŚREDNICH DLA KATEGORII ALL STARS##########
def mean_all_stars_plot(filtered_data):
    plt.figure(figsize=(12, 8))
    all_stars_colors = {0: 'purple', 1: 'blue', 2: 'green', 3: 'red'}
    grouped_data = filtered_data.groupby('All_stars')
    for all_stars_value, group_data in grouped_data:
        mean_stats = group_data.drop(['Player', 'All_stars','MVP'], axis=1).mean()
        stats_index = np.array(mean_stats.index)
        stats_values = mean_stats.values
        plt.plot(stats_index, stats_values, marker='o', linestyle='-', label=f'All Stars {all_stars_value}', color=all_stars_colors[all_stars_value])
    plt.title('Średnie statystyki dla kategorii All Stars')
    plt.xlabel('Kolumna')
    plt.ylabel('Średnia wartość')
    plt.xticks(rotation=45)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
#mean_all_stars_plot(filtered_data0) # 3P 3PA FG% 
#mean_all_stars_plot(filtered_data1) # FG% 2P% eFG% FT% 
#mean_all_stars_plot(filtered_data2) # FG% 2P% eFG% FT%

##########WYRZUCENIE KOLUMN POKRYWAJĄCYCH SIE ZE ŚREDNIMI##########
data_pos0_2023.drop(columns=['3P','3PA','FG%','Sezon','MVP'], inplace=True)
data_pos1_2023.drop(columns=['FG%','2P%','eFG%','FT%','Sezon','MVP'], inplace=True)
data_pos2_2023.drop(columns=['FG%','2P%','eFG%','FT%','Sezon','MVP'], inplace=True)

data_pos0.drop(columns=['3P','3PA','FG%','Sezon','MVP'], inplace=True)
data_pos1.drop(columns=['FG%','2P%','eFG%','FT%','Sezon','MVP'], inplace=True)
data_pos2.drop(columns=['FG%','2P%','eFG%','FT%','Sezon','MVP'], inplace=True)

###########PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY##########
X_train_pos0 = data_pos0.iloc[:,1:-1]
y_train_pos0 = data_pos0.iloc[:,-1]
X_test_pos0 = data_pos0_2023.iloc[:,1:-1]
y_test_pos0 = data_pos0_2023.iloc[:,-1]

X_train_pos1 = data_pos1.iloc[:,1:-1]
y_train_pos1 = data_pos1.iloc[:,-1]
X_test_pos1 = data_pos1_2023.iloc[:,1:-1]
y_test_pos1 = data_pos1_2023.iloc[:,-1]

X_train_pos2 = data_pos2.iloc[:,1:-1]
y_train_pos2 = data_pos2.iloc[:,-1]
X_test_pos2 = data_pos2_2023.iloc[:,1:-1]
y_test_pos2 = data_pos2_2023.iloc[:,-1]

##########USUNIĘCIE NAN##########
nan_indices_train = y_train_pos0[X_train_pos0.isna().any(axis=1)].index
X_train_pos0 = X_train_pos0.dropna()
y_train_pos0 = y_train_pos0.drop(index=nan_indices_train)

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
    selected_indices = np.where((y_pred == 1) | (y_pred == 2) | (y_pred == 3))[0]
    selected_players = data_2023.iloc[selected_indices]['Player']
    #print("Zawodnicy dopasowani do kategorii 1, 2 lub 3:")
    #print(selected_players)
    return selected_players

##########LICZBA WYSTĄPIEŃ DLA KAŻDEGO ZAWODNIKA##########
player_counts0 = {}
player_counts1 = {}
player_counts2 = {}

def player_count(data_pos_2023, X_train_pos, y_train_pos, X_test_pos, player_counts):
    for model in models:
        selected_players = evaluate_model(model, X_train_pos, y_train_pos, X_test_pos, data_pos_2023)
        for player in selected_players:
            if player in player_counts:
                player_counts[player] += 1
            else:
                player_counts[player] = 1
    #print("\nLiczba wystąpień dla każdego zawodnika:")
    #for player, count in player_counts.items():
    #    print(f"{player}: {count}"

player_count(data_pos0_2023, X_train_pos0, y_train_pos0, X_test_pos0, player_counts0)
player_count(data_pos1_2023, X_train_pos1, y_train_pos1, X_test_pos1, player_counts1)
player_count(data_pos2_2023, X_train_pos2, y_train_pos2, X_test_pos2, player_counts2)

##########PODZIAŁ NA ZESPOŁY##########
team1 =[]
team2 =[]
team3 =[]

def add_to_group(player_counts, quantity):
    sorted_players = sorted(player_counts.items(), key=lambda x: x[1], reverse=True)
    if quantity == 1:
        team1.append(sorted_players[0])
        team2.append(sorted_players[1])
        team3.append(sorted_players[2])
    if quantity == 2:
        team1.append(sorted_players[0])
        team1.append(sorted_players[1])
        team2.append(sorted_players[2])
        team2.append(sorted_players[3])
        team3.append(sorted_players[4])
        team3.append(sorted_players[5])

add_to_group(player_counts0, 1)
add_to_group(player_counts1, 2)
add_to_group(player_counts2, 2)

print('Zespół 1')  #Wszyscy prawidłowo 100%
print(team1)
print('Zespół 2')  #2 prawidłowo, 2 powinno być w zespole 3, 1 źle
print(team2)
print('Zespół 3')  #2 prawidłowo, 1 powinno być w zespole 2, 2 źle
print(team3)


###############################################
########## TESTOWANIE MODELU DLA MVP ##########
###############################################

##########PRZYGOTOWANIE DANYCH##########
data_copy = data.copy()
data_2023_copy = data_2023.copy()

data_copy.drop(columns=['Sezon','MVP','FG%','2P%','eFG%','FT%','PF'], inplace=True)
data_2023_copy.drop(columns=['Sezon','MVP','FG%','2P%','eFG%','FT%','PF'], inplace=True)

X_train_data = data_copy.iloc[:,1:-1]
y_train_data = data_copy.iloc[:,-1]
X_test_data_2023 = data_2023_copy.iloc[:,1:-1]
y_test_data_2023 = data_2023_copy.iloc[:,-1]

nan_indices_train = y_train_data[X_train_data.isna().any(axis=1)].index
X_train_data = X_train_data.dropna()
y_train_data = y_train_data.drop(index=nan_indices_train)

nan_indices_train = y_test_data_2023[X_test_data_2023.isna().any(axis=1)].index
X_test_data_2023 = X_test_data_2023.dropna()
y_test_data_2023 = y_test_data_2023.drop(index=nan_indices_train)

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
    selected_indices = np.where((y_pred == 1) | (y_pred == 2) | (y_pred == 3))[0]
    selected_players = data_2023.iloc[selected_indices]['Player']
    #print("Zawodnicy dopasowani do kategorii 1, 2 lub 3:")
    #print(selected_players)
    return selected_players

##########LICZBA WYSTĄPIEŃ DLA KAŻDEGO ZAWODNIKA##########
player_counts = {}

def player_count(data_pos_2023, X_train_pos, y_train_pos, X_test_pos, player_counts):
    for model in models:
        selected_players = evaluate_model(model, X_train_pos, y_train_pos, X_test_pos, data_pos_2023)
        for player in selected_players:
            if player in player_counts:
                player_counts[player] += 1
            else:
                player_counts[player] = 1
    #print("\nLiczba wystąpień dla każdego zawodnika:")
    #for player, count in player_counts.items():
    #    print(f"{player}: {count}"

player_count(data_2023_copy, X_train_data, y_train_data, X_test_data_2023, player_counts)

##########WYBRANIE ZAWODNIKÓW KTÓRZY POWINNI OTRZYMAĆ NAGRODĘ MVP##########
MVP_avard = []
for player, count in player_counts.items():
    if count == 9:
        MVP_avard.append(player)

##########WYKRESY DLA ZAWODNIKÓW MVP##########
selected_players_data = data_2023_copy[data_2023_copy['Player'].isin(MVP_avard)]
selected_players_data = pd.DataFrame(selected_players_data)

def player_stats_plot(selected_players_data):
    plt.figure(figsize=(12, 8))
    for idx, row in selected_players_data.iterrows():
        player_name = row['Player']
        player_data = row.drop(['Player', 'Pos', 'All_stars'])
        plt.plot(player_data, marker='o', label=player_name)
    plt.title('Statystyki dla wybranych zawodników')
    plt.xlabel('Indeks')
    plt.ylabel('Wartość')
    plt.xticks(rotation=45)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#player_stats_plot(selected_players_data)

##########ŚREDNIE STATYSTYKI DLA ZAWODNIKÓW MVP##########
mean_pos_0 = data_2023[data_2023['Pos'] == 0]
mean_pos_0 = mean_pos_0.drop(columns=['Player','Sezon','MVP','FG%','2P%','eFG%','FT%','PF','Pos', 'All_stars'])
mean_pos_0 = mean_pos_0.mean()

mean_pos_1 = data_2023[data_2023['Pos'] == 1]
mean_pos_1 = mean_pos_1.drop(columns=['Player','Sezon','MVP','FG%','2P%','eFG%','FT%','PF','Pos', 'All_stars'])
mean_pos_1 = mean_pos_1.mean()

mean_pos_2 = data_2023[data_2023['Pos'] == 2]
mean_pos_2 = mean_pos_2.drop(columns=['Player','Sezon','MVP','FG%','2P%','eFG%','FT%','PF','Pos', 'All_stars'])
mean_pos_2 = mean_pos_2.mean()

selected_players_data_pos_0 = selected_players_data[selected_players_data['Pos'] == 0]
selected_players_data_pos_1 = selected_players_data[selected_players_data['Pos'] == 1]
selected_players_data_pos_2 = selected_players_data[selected_players_data['Pos'] == 2]

##########WYBÓR MVP##########
diff_pos_0 = selected_players_data_pos_0.drop(columns=['Player', 'Pos', 'All_stars']).sub(mean_pos_0)
diff_pos_1 = selected_players_data_pos_1.drop(columns=['Player', 'Pos', 'All_stars']).sub(mean_pos_1)
diff_pos_2 = selected_players_data_pos_2.drop(columns=['Player', 'Pos', 'All_stars']).sub(mean_pos_2)

all_diff = pd.concat([diff_pos_0, diff_pos_1, diff_pos_2])
sum_diff_all = all_diff.sum(axis=1)
max_diff_index = sum_diff_all.idxmax()
max_diff_player_name = selected_players_data.loc[max_diff_index, 'Player']

print("MVP to: ", max_diff_player_name)
