#necessario aver installato numpy e pandas per utilizzare questo generatore
from random import seed
from random import random
from random import randint 
import pandas as pd
import numpy as np
# seed 
seed()
df = pd.DataFrame(columns=range(0,31))
print("Inizio generazione training")
for i in range (0,64000):
	tmp_list = []
	for j in range(0, 30):
		value = random()
		#print(value)
		roundValue = np.around(value, decimals = 6) 
		#print(roundValue)
		tmp_list.append(roundValue)
	#print(tmp_list)
	#print(round_tmp)
	label = randint(0, 9)
	tmp_list.append(label)
	df.loc[i] = tmp_list

df.to_csv("./train", index=False, header = False, sep=" ")

print("Inizio generazione test")
seed(1)
df = pd.DataFrame(columns=range(0,31))
for i in range (0,16000):
	tmp_list = []
	for j in range(0, 30):
		value = random()
		#print(value)
		roundValue = np.around(value, decimals = 6) 
		#print(roundValue)
		tmp_list.append(roundValue)
	#print(tmp_list)
	#print(round_tmp)
	label = randint(0, 9)
	#print(label)
	tmp_list.append(label)
	df.loc[i] = tmp_list

df.to_csv("./test", index=False, header = False, sep=" ")
