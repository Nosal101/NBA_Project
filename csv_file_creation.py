import pandas as pd

########DANE##########
nba16 = 'Data/nba16.csv'
nba17 = 'Data/nba17.csv'
nba18 = 'Data/nba18.csv'
nba19 = 'Data/nba19.csv'
nba20 = 'Data/nba20.csv'
nba21 = 'Data/nba21.csv'
nba22 = 'Data/nba22.csv'
nba23 = 'Data/nba23.csv'
nba24 = 'Data/nba24.csv'

All_stars16_1 = ['Kawhi Leonard', 'LeBron James', 'DeAndre Jordan', 'Stephen Curry', 'Russell Westbrook']
All_stars16_2 = ['Kevin Durant', 'Draymond Green', 'DeMarcus Cousins', 'Damian Lillard', 'Chris Paul']
All_stars16_3 = ['Paul George', 'LaMarcus Aldridge', 'Andre Drummond', 'Klay Thompson', 'Kyle Lowry']

All_stars17_1 = ['Kawhi Leonard', 'LeBron James', 'Anthony Davis', 'James Harden', 'Russell Westbrook']
All_stars17_2 = ['Kevin Durant', 'Giannis Antetokounmpo', 'Rudy Gobert', 'Stephen Curry', 'Isaiah Thomas']
All_stars17_3 = ['Jimmy Butler','Draymond Green', 'DeAndre Jordan', 'John Wall', 'DeMar DeRozan']

All_stars18_1 = ['Kevin Durant', 'LeBron James', 'Anthony Davis', 'James Harden', 'Damian Lillard']
All_stars18_2 = ['LaMarcus Aldridge', 'Giannis Antetokounmpo', 'Joel Embiid', 'DeMar DeRozan', 'Russell Westbrook']
All_stars18_3 = ['Jimmy Butler', 'Paul George', 'Karl-Anthony Towns', 'Victor Oladipo', 'Stephen Curry']

All_stars19_1 = ['Giannis Antetokounmpo', 'Paul George', 'Nikola Jokić', 'James Harden', 'Stephen Curry']
All_stars19_2 = ['Kevin Durant', 'Kawhi Leonard', 'Joel Embiid', 'Damian Lillard', 'Kyrie Irving']
All_stars19_3 = ["Blake Griffin",'LeBron James', 'Rudy Gobert', 'Russell Westbrook', 'Kemba Walker']

All_stars20_1 = ['LeBron James', 'Giannis Antetokounmpo', 'Anthony Davis', 'James Harden', 'Luka Dončić']
All_stars20_2 = ['Kawhi Leonard', 'Pascal Siakam', 'Nikola Jokić', 'Damian Lillard', 'Chris Paul']
All_stars20_3 = ['Jimmy Butler', 'Jayson Tatum', 'Rudy Gobert', 'Ben Simmons', 'Russell Westbrook']

All_stars21_1 = ['Giannis Antetokounmpo', 'Kawhi Leonard', 'Nikola Jokić', 'Stephen Curry', 'Luka Dončić']
All_stars21_2 = ['LeBron James', 'Julius Randle', 'Joel Embiid', 'Damian Lillard', 'Chris Paul']
All_stars21_3 = ['Jimmy Butler', 'Paul George', 'Rudy Gobert', 'Bradley Beal', 'Kyrie Irving']

All_stars22_1 = ['Giannis Antetokounmpo', 'Jayson Tatum', 'Nikola Jokić', 'Devin Booker', 'Luka Dončić']
All_stars22_2 = ['DeMar DeRozan', 'Kevin Durant', 'Joel Embiid', 'Stephen Curry', 'Ja Morant']
All_stars22_3 = ['LeBron James', 'Pascal Siakam', 'Karl-Anthony Towns', 'Chris Paul', 'Trae Young']

All_stars23_1 = ['Giannis Antetokounmpo', 'Jayson Tatum', 'Joel Embiid', 'Luka Dončić', 'Shai Gilgeous-Alexander']
All_stars23_2 = ['Jimmy Butler', 'Jaylen Brown', 'Nikola Jokić', 'Stephen Curry', 'Donovan Mitchell']
All_stars23_3 = ['LeBron James', 'Julius Randle', 'Domantas Sabonis', 'De Aaron Fox', 'Damian Lillard']

MVP16 = 'Stephen Curry'
MVP17 = 'Russell Westbrook'
MVP18 = 'James Harden'
MVP19 = "Giannis Antetokounmpo"
MVP20 = "Giannis Antetokounmpo"
MVP21 = 'Nikola Jokić'
MVP22 = 'Nikola Jokić'
MVP23 = 'Joel Embiid'

########WCZYTANIE DANYCH##########
nba16 = pd.read_csv(nba16).iloc[:, 1:-1]
nba17 = pd.read_csv(nba17).iloc[:, 1:-1]
nba18 = pd.read_csv(nba18).iloc[:, 1:-1]
nba19 = pd.read_csv(nba19).iloc[:, 1:-1]
nba20 = pd.read_csv(nba20).iloc[:, 1:-1]
nba21 = pd.read_csv(nba21).iloc[:, 1:-1]
nba22 = pd.read_csv(nba22).iloc[:, 1:-1]
nba23 = pd.read_csv(nba23).iloc[:, 1:-1]
nba24 = pd.read_csv(nba24).iloc[:, 1:-1]

########SORTOWANIE I USUWANIE DUPLIKATÓW##########
nazwy_plikow = [nba16, nba17, nba18, nba19, nba20, nba21, nba22, nba23, nba24]
ramki_danych = []

def Sorted_Uniq_data(df):
  df = df.sort_values(by=['Player', 'G'], ascending=[True, False])
  df = df.drop_duplicates(subset=['Player'], keep='first')
  return df

for nazwa_pliku in nazwy_plikow:
    ramki_danych.append(Sorted_Uniq_data(nazwa_pliku))

nba16, nba17, nba18, nba19, nba20, nba21, nba22, nba23, nba24 = ramki_danych

########DODANIE KOLUMNY ALL_STARS##########
def add_all_stars_column(df, all_stars_1, all_stars_2, all_stars_3):
    df['All_stars'] = 0
    for index, row in df.iterrows():
        player = row['Player']

        if player in all_stars_1:
            df.at[index, 'All_stars'] = 1
        elif player in all_stars_2:
            df.at[index, 'All_stars'] = 2
        elif player in all_stars_3:
            df.at[index, 'All_stars'] = 3
def count_all_stars(df,name):
    if 'All_stars' in df.columns:
        counts = df['All_stars'].value_counts()
        print(counts)
    else:
        print("Kolumna 'All_stars' nie istnieje w DataFrame'ie.")

add_all_stars_column(nba16,All_stars16_1,All_stars16_2,All_stars16_3)
add_all_stars_column(nba17,All_stars17_1,All_stars17_2,All_stars17_3)
add_all_stars_column(nba18,All_stars18_1,All_stars18_2,All_stars18_3)
add_all_stars_column(nba19,All_stars19_1,All_stars19_2,All_stars19_3)
add_all_stars_column(nba20,All_stars20_1,All_stars20_2,All_stars20_3)
add_all_stars_column(nba21,All_stars21_1,All_stars21_2,All_stars21_3)
add_all_stars_column(nba22,All_stars22_1,All_stars22_2,All_stars22_3)
add_all_stars_column(nba23,All_stars23_1,All_stars23_2,All_stars23_3)
nba24['All_stars'] = 'NaN'

########DODANIE KOLUMNY MVP##########
def add_mvp_column(df,mvp):
    df['MVP'] = 0
    for index, row in df.iterrows():
        player = row['Player']
        if player in mvp:
            df.at[index, 'MVP'] = 1
def count_mvp(df,name):
    if 'MVP' in df.columns:
        counts = df['MVP'].value_counts()
        print(counts)
    else:
        print("Kolumna 'MVP' nie istnieje w DataFrame")

add_mvp_column(nba16,MVP16)
add_mvp_column(nba17,MVP17)
add_mvp_column(nba18,MVP18)
add_mvp_column(nba19,MVP19)
add_mvp_column(nba20,MVP20)
add_mvp_column(nba21,MVP21)
add_mvp_column(nba22,MVP22)
add_mvp_column(nba23,MVP23)
nba24['MVP'] = 'NaN'

########DODANIE KOLUMNY SEZON##########
nba16['Sezon'] = 2016
nba17['Sezon'] = 2017
nba18['Sezon'] = 2018
nba19['Sezon'] = 2019
nba20['Sezon'] = 2020
nba21['Sezon'] = 2021
nba22['Sezon'] = 2022
nba23['Sezon'] = 2023
nba24['Sezon'] = 2024

########SCALANIE DANYCH##########
nba = pd.concat([nba16, nba17, nba18, nba19, nba20, nba21, nba22, nba23,nba24])

########ZAPIS DO PLIKU##########
nba.to_csv('połączony_nba.csv', index=False)
