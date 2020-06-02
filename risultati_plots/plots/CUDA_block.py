from pylab import * 
from numpy import *

if __name__ == '__main__':
    test = [480 ,800, 8000, 16000, 32000, 64000]
    NT = [8, 16, 32]
    CUDA = {8: [0.011303, 0.021059, 0.202351, 0.432082, 0.913778, 2.230188], 16: [0.011685, 0.021461, 0.199183, 0.442126, 0.733868, 2.449643], 32: [0.011796, 0.016685, 0.206116, 0.43993, 0.996463, 2.921638]}

    colors = "cyan-dodgerblue-blue".split('-')
    figure(figsize=(8,6))
    distanze = [[0.25, 15.75, 31.25, 46.75, 62.25, 77.75],
[2.25, 17.75, 33.25, 48.75, 64.25, 79.75],
[4.25, 19.75, 35.25, 50.75, 66.25, 81.75]]


    bar(distanze[0], CUDA[8] , width=1.5, color=colors[0], ec="black", label="8x8")
    bar(distanze[1] , CUDA[16] , width=1.5,  color=colors[1], ec="black", label="16x16")
    bar(distanze[2], CUDA[32] , width=1.5, color=colors[2], ec="black", label="32x32")


    '''
    for i in range(6):
        c = 0.25;
        c = c+ i * 2;
        print("[", end = '')
        for j in range(8):
            print(str(c) + ", ", end = '')
            c = c + 15.5
        print("],")
    '''
    
    xticks([2.25, 17.75, 33.25, 48.75, 64.25, 79.75], test)
    title("Confronto tempo al variare del numero di thread per blocco" , fontsize=16)
    yscale('log')
    xlabel("Dimensione dataset", fontsize=14)
    ylabel("Tempo richiesto", fontsize=14)

    legend()
    savefig("threadCUDA.pdf")
    close()


    #test secondo dataset
    test = ["30000 x 784"]
    NT = [8, 16, 32]
    CUDA = {8: [26.921494], 16: [23.299089], 32: [24.082211]}

    colors = "cyan-dodgerblue-blue".split('-')
    figure(figsize=(8,6))
    distanze = [[0.25], [2.25], [4.25]]


    bar(distanze[0], CUDA[8] , width=1.5, color=colors[0], ec="black", label="8x8")
    bar(distanze[1] , CUDA[16] , width=1.5,  color=colors[1], ec="black", label="16x16")
    bar(distanze[2], CUDA[32] , width=1.5, color=colors[2], ec="black", label="32x32")


    #xticks([5], test)
    title("Confronto tempo al variare del numero di thread per blocco dataset 30000 x 784", fontsize=12)
    #yscale('log')
    #xlabel("Dimensione dataset", fontsize=14)
    ylabel("Tempo richiesto", fontsize=14)
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    legend()
    savefig("threadCUDA_attr.pdf")
    close()
