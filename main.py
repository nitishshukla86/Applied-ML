from prepare_data import get_data,get_data_SVM
from models import lstm,support_vector_clf
import matplotlib.pyplot as plt
from visualize_data import plot_trend
from score_metrics import plot_confusion,classification_rep
import argparse
import numpy as np

def main(args):
    if args.model=='lstm':
        n,new_data,scaler,x_train_open,x_train_close,y_train_open,y_train_close,X_test_open,X_test_close=\
            get_data(name='AAPL',start=args.start, end=args.end)
        print("Start training on closing prices")
        model=lstm((x_train_close.shape[1],1))
        model.fit(x_train_close, y_train_close, epochs=args.epochs, batch_size=2, verbose=1)
        clo_price = model.predict(X_test_close)
        print("Train finished!!\n\n")

        print("Start training on opening prices")
        model=lstm((x_train_open.shape[1],1))
        model.fit(x_train_open, y_train_open, epochs=args.epochs, batch_size=2, verbose=1)
        open_price=model.predict(X_test_open)
        print("Train finished!!\n\n")

        closing_price=scaler.inverse_transform(np.stack((open_price[:,0],clo_price[:,0]),axis=1))[:,0]
        opening_price=scaler.inverse_transform(np.stack((open_price[:,0],clo_price[:,0]),axis=1))[:,1]
        plot_trend(n,new_data,opening_price,key="Open",save=args.save)
        plot_trend(n,new_data,opening_price,key="Close",save=args.save)

        ##Binarizing the predictions
        y_pred=((closing_price-opening_price)>0).astype(int)
        y_true=((new_data[n:]['Close']-new_data[n:]['Open'])>0).astype(int)

        #from sklearn.metrics import confusion_matrix
        plot_confusion(y_pred,y_true,save=args.save,name="confusion_lstm")

        #classification report
        print("Classification report:")
        print(classification_rep(y_pred,y_true))
        
        
    elif args.model=='svc':
        df,X_train,X_test,y_train,y_test=get_data_SVM()
        cls=support_vector_clf(X_train,y_train)
        y_pred=cls.predict(X_test)
        plot_confusion(y_pred,y_test,save=True,name="confusion_svc")
        print("Classification report:")
        print(classification_rep(y_pred,y_test))
        
        df['Predicted_Signal'] = cls.predict(np.concatenate([X_train,X_test]))
        df['Return'] = df.Close.pct_change()
        df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)
        df['Cum_Ret'] = df['Return'].cumsum()
        df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
        plt.figure()
        plt.plot(df['Cum_Ret'],color='red')
        plt.plot(df['Cum_Strategy'],color='blue')
        plt.title('Predicted Returns vs Original Returns')
        plt.savefig('results/SVC.png')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="select model lstm/scv",
                    default="svc")
    
    parser.add_argument("--epochs", help="number of epochs",
                    default=2,type=int)
    parser.add_argument("--start", help="start date",
                    default='2020-01-01')
    parser.add_argument("--end", help="end date",
                    default='2021-06-12')
    parser.add_argument("--save", help="save results",
                    default=True)
    
    args=parser.parse_args()
    main(args)
    



    
        
    
