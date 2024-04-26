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

data_pos0_2023_copy = data_pos0_2023.copy()
data_pos1_2023_copy = data_pos1_2023.copy()
data_pos2_2023_copy = data_pos2_2023.copy()

data_pos0_2023_copy.drop(columns=['Sezon','All_stars','MVP'], inplace=True)
data_pos1_2023_copy.drop(columns=['Sezon','All_stars','MVP'], inplace=True)
data_pos2_2023_copy.drop(columns=['Sezon','All_stars','MVP'], inplace=True)

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

##########REGRESJA LOGISTYCZNA##########
model = LogisticRegression(random_state=0)
model.fit(X_train_pos0, y_train_pos0)
y_pred = model.predict(X_test_pos0)
selected_indices = np.where((y_pred == 1) | (y_pred == 2) | (y_pred == 3))[0]
selected_players = data_pos0_2023.iloc[selected_indices]['Player']
print("Zawodnicy dopasowani do kategorii 1, 2 lub 3: ")
print(selected_players)


model = LogisticRegression(random_state=0)
model.fit(X_train_pos1, y_train_pos1)
y_pred = model.predict(X_test_pos1)
selected_indices = np.where((y_pred == 1) | (y_pred == 2) | (y_pred == 3))[0]
selected_players = data_pos1_2023.iloc[selected_indices]['Player']
print("Zawodnicy dopasowani do kategorii 1, 2 lub 3:")
print(selected_players)


model = LogisticRegression(random_state=0)
model.fit(X_train_pos2, y_train_pos2)
y_pred = model.predict(X_test_pos2)
selected_indices = np.where((y_pred == 1) | (y_pred == 2) | (y_pred == 3))[0]
selected_players = data_pos2_2023.iloc[selected_indices]['Player']
print("Zawodnicy dopasowani do kategorii 1, 2 lub 3:")
print(selected_players)


# Dokładość Modelu 87%

























# def calculate_mean_stats(filtered_data):
#     all_stars_mean_stats = {}
#     grouped_data = filtered_data.groupby('All_stars')
#     for all_stars_value, group_data in grouped_data:
#         mean_stats = group_data.drop(['Player', 'All_stars','MVP'], axis=1).mean()
#         all_stars_mean_stats[all_stars_value] = mean_stats
#     return all_stars_mean_stats

# mean_data0 = calculate_mean_stats(filtered_data0)[1]
# mean_data1 = calculate_mean_stats(filtered_data1)[1]
# mean_data2 = calculate_mean_stats(filtered_data2)[1]

#print(mean_data0.shape[0])
#print(data_pos0_2023)


























# def calculate_total_deviation(data_2023, mean_data):
#     total_deviations = []
#     for idx, row in data_2023.iterrows():
#         player_name = row['Player']
#         player_stats = row.drop(['Player'])  # Usunięcie kolumny z imieniem zawodnika i rokiem
#         total_deviation = np.abs(player_stats - mean_data).sum()  # Obliczanie łącznego odchylenia
#         total_deviations.append((player_name, total_deviation))
#     return total_deviations

# total_deviations_2023 = calculate_total_deviation(data_pos0_2023, mean_data0)
# sorted_deviations = sorted(total_deviations_2023, key=lambda x: x[1])
# # Wyświetlenie trzech zawodników z najmniejszymi różnicami
# print("Trzy zawodniki z najmniejszymi różnicami od średnich statystyk:")
# for player, deviation in sorted_deviations[:3]:
#     print(f"{player}: {deviation}")


# total_deviations_2023 = calculate_total_deviation(data_pos1_2023, mean_data1)
# sorted_deviations = sorted(total_deviations_2023, key=lambda x: x[1])
# # Wyświetlenie trzech zawodników z najmniejszymi różnicami
# print("Trzy zawodniki z najmniejszymi różnicami od średnich statystyk:")
# for player, deviation in sorted_deviations[:6]:
#     print(f"{player}: {deviation}")


# total_deviations_2023 = calculate_total_deviation(data_pos2_2023, mean_data2)
# sorted_deviations = sorted(total_deviations_2023, key=lambda x: x[1])
# # Wyświetlenie trzech zawodników z najmniejszymi różnicami
# print("Trzy zawodniki z najmniejszymi różnicami od średnich statystyk:")
# for player, deviation in sorted_deviations[:6]:
#     print(f"{player}: {deviation}")









































######################################################3
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# names = data['Player']
# data = data.drop(['Player', 'Age', 'Tm', '2P%', '3P%'], axis=1)

# #Podział na Centrów Obrońców Skrzydłowych
# data_pos0 = data[data['Pos'] == 0]
# data_pos1 = data[data['Pos'] == 1]
# data_pos2 = data[data['Pos'] == 2]

# # Podział danych na zestaw treningowy i testowy
# train_data_pos0 = data_pos0[data_pos0['Sezon'].isin(range(2016, 2023))]
# test_data_pos0 = data_pos0[data_pos0['Sezon'] == 2023]

# train_data_pos1 = data_pos1[data_pos1['Sezon'].isin(range(2016, 2023))]
# test_data_pos1 = data_pos1[data_pos1['Sezon'] == 2023]  

# train_data_pos2 = data_pos2[data_pos2['Sezon'].isin(range(2016, 2023))]
# test_data_pos2 = data_pos2[data_pos2['Sezon'] == 2023]


# X_train_pos0 = train_data_pos0.drop('All_stars', axis=1)
# y_train_pos0 = train_data_pos0['All_stars']
# X_test_pos0 = test_data_pos0.drop('All_stars', axis=1)
# y_test_pos0 = test_data_pos0['All_stars']

# X_train_pos1 = train_data_pos1.drop('All_stars', axis=1)
# y_train_pos1 = train_data_pos1['All_stars']
# X_test_pos1 = test_data_pos1.drop('All_stars', axis=1)
# y_test_pos1 = test_data_pos1['All_stars']

# X_train_pos2 = train_data_pos2.drop('All_stars', axis=1)
# y_train_pos2 = train_data_pos2['All_stars']
# X_test_pos2 = test_data_pos2.drop('All_stars', axis=1)
# y_test_pos2 = test_data_pos2['All_stars']

# # Inicjalizacja i trening modelu
# clf0 = RandomForestClassifier()
# clf0.fit(X_train_pos0, y_train_pos0)

# clf1 = RandomForestClassifier()
# clf1.fit(X_train_pos1, y_train_pos1)

# clf2 = RandomForestClassifier()
# clf2.fit(X_train_pos2, y_train_pos2)

# # Predykcja na danych testowych
# y_pred0 = clf0.predict(X_test_pos0)
# y_pred1 = clf1.predict(X_test_pos1)
# y_pred2 = clf2.predict(X_test_pos2)

# #Wynik
# selected_players_indices0 = test_data_pos0.index[y_pred0 != 0][:5]
# selected_players0 = names.loc[selected_players_indices0]
# print(selected_players0)

# selected_players_indices1 = test_data_pos1.index[y_pred1 != 0][:5]
# selected_players1 = names.loc[selected_players_indices1]
# print(selected_players1)

# selected_players_indices2 = test_data_pos2.index[y_pred2 != 0][:5]
# selected_players2 = names.loc[selected_players_indices2]
# print(selected_players2)

