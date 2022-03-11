import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def plot_confusion(y_pred,y_true,save=True,name=None):
    mat=confusion_matrix(y_true,y_pred)
    df_cm = pd.DataFrame(mat, index = [i for i in "01"],
                  columns = [i for i in "01"])
    plt.figure(figsize = (10,7))
    plt.title('Confusion matrix')
    sn.heatmap(df_cm, annot=True)
    if save:
        svm=sn.heatmap(df_cm, annot=True)
        figure = svm.get_figure()    
        figure.savefig(f'results/{name}.png', dpi=400)
 


def classification_rep(y_pred,y_true):
    return classification_report(y_true,y_pred)

    
