import matplotlib.pyplot as plt

def plotmultileadSig(sigmas):
    plt.figure()
    counter = 0
    for leadsig in sigmas:
        counter+=1
        plt.subplot(len(sigmas),1,counter)
        plt.plot(leadsig)
    plt.show()

