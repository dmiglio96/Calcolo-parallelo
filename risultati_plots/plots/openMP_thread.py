from pylab import * 
from numpy import *

if __name__ == '__main__':
    test = [480 ,800, 8000, 16000, 32000, 64000]
    NT = [2, 4, 8, 16, 32]
    openMP = {2: [0.01211, 0.052862, 1.697478, 4.040084, 16.752405, 56.825848], 4: [0.005238, 0.016425, 1.02735, 2.8206, 8.252842, 31.092289], 8: [0.004482, 0.01505, 0.678595, 1.97765, 5.219484, 16.260672], 16: [0.005187, 0.024208, 0.711001, 1.515463, 3.307396, 9.554554], 32: [0.010835, 0.015484, 0.318661, 1.335839, 2.950037, 9.372705]}
    colors = "cyan-darkturquoise-deepskyblue-dodgerblue-royalblue-blue".split('-')

    figure(figsize=(8,6))
    distanze = [[0.25, 15.75, 31.25, 46.75, 62.25, 77.75],
[2.25, 17.75, 33.25, 48.75, 64.25, 79.75],
[4.25, 19.75, 35.25, 50.75, 66.25, 81.75],
[6.25, 21.75, 37.25, 52.75, 68.25, 83.75],
[8.25, 23.75, 39.25, 54.75, 70.25, 85.75],
[10.25, 25.75, 41.25, 56.75, 72.25, 87.75]]

    #bar(distanze[0], openMP[1] , width=1.5, color=colors[0], ec="black", label="1 Thread")
    bar(distanze[0] , openMP[2] , width=1.5,  color=colors[1], ec="black", label="2 Thread")
    bar(distanze[1], openMP[4] , width=1.5, color=colors[2], ec="black", label="4 Thread")
    bar(distanze[2]  , openMP[8] , width=1.5, color=colors[3], ec="black",  label="8 Thread")
    bar(distanze[3], openMP[16] , width=1.5,  color=colors[4], ec="black", label="16 Thread")
    bar(distanze[4] , openMP[32] , width=1.5, color=colors[5], ec="black", label="32 Thread")    
    
    xticks([5, 20.5, 36, 51.5, 67, 82.5], test)
    title("Confronto tempo al variare del numero di thread", fontsize=16)
    yscale('log')
    xlabel("Dimensione dataset", fontsize=14)
    ylabel("Tempo richiesto", fontsize=14)

    legend()
    savefig("threadOpenMP.pdf")
    close()





    test = ["30000 x 784"]
    NT = [2, 4, 8, 16, 32]
    openMP = {2: [765.130981], 4: [391.633667], 8: [208.888092], 16: [115.306534], 32: [117.969971]}
    colors = "cyan-darkturquoise-deepskyblue-dodgerblue-royalblue-blue".split('-')

    figure(figsize=(8,6))
    distanze = [[0.25], [2.25], [4.25], [6.25], [8.25], [10.25]]

    #bar(distanze[0], openMP[1] , width=1.5, color=colors[0], ec="black", label="1 Thread")
    bar(distanze[0] , openMP[2] , width=1.5,  color=colors[1], ec="black", label="2 Thread")
    bar(distanze[1], openMP[4] , width=1.5, color=colors[2], ec="black", label="4 Thread")
    bar(distanze[2]  , openMP[8] , width=1.5, color=colors[3], ec="black",  label="8 Thread")
    bar(distanze[3], openMP[16] , width=1.5,  color=colors[4], ec="black", label="16 Thread")
    bar(distanze[4] , openMP[32] , width=1.5, color=colors[5], ec="black", label="32 Thread")    
    
    xticks([5], test)
    title("Confronto tempo al variare del numero di thread 30000 x 784", fontsize=12)
    yscale('log')
    #xlabel("Dimensione dataset", fontsize=14)
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    ylabel("Tempo richiesto", fontsize=14)

    legend()
    savefig("threadOpenMP_attr.pdf")
    close()
