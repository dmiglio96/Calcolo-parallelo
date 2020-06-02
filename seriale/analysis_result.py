from statistics import * 
medTime = []
stdTime = []
npixel = []

with open("resultsLBP_serial.out", "r") as f:
	fl = f.readlines()
	time = []
	count = 0
	for x in fl:
		l = []
		for t in x.split():			#ogni esperimento
			try:
				l.append(float(t))	
			except ValueError:
				pass
		#print(x)
		#print(l)
		time.append(l[2])
		#print(count)
		#print(l)
		count = count + 1
		

		if count == 10:
			medTime.append(mean(time))
			stdTime.append(stdev(time))
			npixel.append(int(l[0]) * int(l[1]))
			time = []
			count = 0
	
	print("tempi medi", medTime)
	print("dev stand", stdTime)
	print("npixel", npixel)

		