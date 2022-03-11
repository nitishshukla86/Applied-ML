import matplotlib.pyplot as plt
def plot_trend(n,new_data,pred,key="Close",save=True):
    train = new_data[:n]
    valid = new_data[n:]
    valid['Predictions'] = pred
    fig=plt.figure()
    plt.plot(train[key])
    plt.plot(valid[key],label="Actual")
    plt.plot(valid['Predictions'],label="Predicted")
    plt.title(f'{key} predictions')
    plt.legend()
    if save:
        plt.savefig(f"results/{key}_price.png")
    fig.show()
    