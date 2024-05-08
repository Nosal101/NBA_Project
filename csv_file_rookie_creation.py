import pandas as pd

#############DANE################
nba16 = 'Data/rookie_16.csv'
nba17 = 'Data/rookie_17.csv'
nba18 = 'Data/rookie_18.csv'
nba19 = 'Data/rookie_19.csv'
nba20 = 'Data/rookie_20.csv'
nba21 = 'Data/rookie_21.csv'
nba22 = 'Data/rookie_22.csv'
nba23 = 'Data/rookie_23.csv'
nba24 = 'Data/rookie_24.csv'

Rookies_16_1 = ['Karl-Anthony Towns', 'Kristaps Porzingis', 'Devin Booker', 'Nikola Jokic', 'Jahlil Okafor']
Rookies_16_2 = ['Justise Winslow', 'D Angelo Russell', 'Emmanuel Mudiay', 'Myles Turner', 'Willie Cauley-Stein']

Rookies_17_1 = ['Malcolm Brogdon', 'Dario Sarić', 'Joel Embiid', 'Buddy Hield', 'Willy Hernangómez']
Rookies_17_2 = ['Jamal Murray', 'Jaylen Brown', 'Marquese Chriss', 'Brandon Ingram', 'Yogi Ferrell']

Rookies_18_1 = ['Ben Simmons', 'Donovan Mitchell', 'Jayson Tatum', 'Kyle Kuzma', 'Lauri Markkanen']
Rookies_18_2 = ['Dennis Smith Jr.', 'Lonzo Ball', 'John Collins', 'Bogdan Bogdanovic', 'Josh Jackson']

Rookies_19_1 = ['Luka Doncic', 'Trae Young', 'Deandre Ayton', 'Jaren Jackson Jr.', 'Marvin Bagley III']
Rookies_19_2 = ['Shai Gilgeous-Alexander', 'Collin Sexton', 'Landry Shamet', 'Mitchell Robinson', 'Kevin Huerter']

Rookies_20_1 = ['Ja Morant', 'Kendrick Nunn', 'Brandon Clarke', 'Zion Williamson', 'Eric Paschall']
Rookies_20_2 = ['Tyler Herro', 'Terence Davis', 'Coby White', 'P.J. Washington', 'Rui Hachimura']

Rookies_21_1 = ['LaMelo Ball', 'Anthony Edwards', 'Tyrese Haliburton', 'Saddiq Bey', 'Jae Sean Tate']
Rookies_21_2 = ['Immanuel Quickley', 'Desmond Bane', 'Isaiah Stewart', 'Isaac Okoro', 'Patrick Williams']

Rookies_22_1 = ['Scottie Barnes', 'Evan Mobley', 'Cade Cunningham', 'Franz Wagner', 'Jalen Green']
Rookies_22_2 = ['Herbert Jones', 'Chris Duarte', 'Bones Hyland', 'Ayo Dosunmu', 'Josh Giddey']

Rookies_23_1 = ['Paolo Banchero', 'Walker Kessler', 'Bennedict Mathurin', 'Keegan Murray', 'Jalen Williams']
Rookies_23_2 = ['Jalen Duren', 'Jaden Ivey', 'Jabari Smith Jr.', 'Tari Eason', 'Jeremy Sochan']

Rookies_16_MVP = 'Karl-Anthony Towns'
Rookies_17_MVP = 'Malcolm Brogdon'
Rookies_18_MVP = 'Ben Simmons'
Rookies_19_MVP = 'Luka Dončić'
Rookies_20_MVP = 'Ja Morant'
Rookies_21_MVP = 'LaMelo Ball'
Rookies_22_MVP = 'Scottie Barnes'
Rookies_23_MVP = 'Paolo Banchero'


########WCZYTANIE DANYCH##########
nba16 = pd.read_csv(nba16)
nba17 = pd.read_csv(nba17)
nba18 = pd.read_csv(nba18)
nba19 = pd.read_csv(nba19)
nba20 = pd.read_csv(nba20)
nba21 = pd.read_csv(nba21)
nba22 = pd.read_csv(nba22)
nba23 = pd.read_csv(nba23)
nba24 = pd.read_csv(nba24)

########DODANIE KOLUMNY ALL_STARS##########
def add_all_stars_column(df, all_stars_1, all_stars_2):
    df['All_stars'] = 0
    for index, row in df.iterrows():
        player = row['PLAYER']

        if player in all_stars_1:
            df.at[index, 'All_stars'] = 1
        elif player in all_stars_2:
            df.at[index, 'All_stars'] = 2

def count_all_stars(df,name):
    if 'All_stars' in df.columns:
        counts = df['All_stars'].value_counts()
        print(counts)
        print(df[df['All_stars'] == 1])
        
    else:
        print("Kolumna 'All_stars' nie istnieje w DataFrame'ie.")

add_all_stars_column(nba16, Rookies_16_1, Rookies_16_2)
add_all_stars_column(nba17, Rookies_17_1, Rookies_17_2)
add_all_stars_column(nba18, Rookies_18_1, Rookies_18_2)
add_all_stars_column(nba19, Rookies_19_1, Rookies_19_2)
add_all_stars_column(nba20, Rookies_20_1, Rookies_20_2)
add_all_stars_column(nba21, Rookies_21_1, Rookies_21_2)
add_all_stars_column(nba22, Rookies_22_1, Rookies_22_2)
add_all_stars_column(nba23, Rookies_23_1, Rookies_23_2)
nba24['All_stars'] = 'NaN'

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

print(nba24)
print(nba22)
########ZAPIS DO PLIKU##########
nba.to_csv('połączony_rookie_nba.csv', index=False)