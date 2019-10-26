import pandas as pd
import random as rand

df = pd.read_csv('students.csv', )

population = list(df['name'])

teams = []

while len(population) >4:
    team = rand.sample(population, k=3)
    population = [x for x in population if x not in team]
    teams.append(team)
teams.append(population)

team_leaders = []

for team in teams:
    team_leader = rand.sample(team, k=1)
    team_leaders.append(team_leader)

print(teams)
print(team_leaders)

print(df)