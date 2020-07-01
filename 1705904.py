### TODO: A utility class which will be created by the user , with Class name as " _[your roll number]" ,
# TODO: all transformations should be written inside a function which will be called inside the predict method
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder

class _1705904():

    ## TODO: Please note that document id should be present till the getPredictions method
    def __tranformation1(self,data):

        # your transformation logic goes here
        df = data
        mapper=df.groupby('document_number')['invoice_amount_doc_currency'].mean().to_dict()
        df['invoice_currency_avg']=df['document_number'].map(mapper)
        return df

    def __transformation2(self,data):

        # your transformation logic goes here
        return data

    def getPredictions(self,data,model):
        temp = data
        data = self.__tranformation1(data)
        data = self.__transformation2(data)
        # your feature list, column names
        features = ['acct_doc_header_id','customer_number','cust_payment_terms','dayspast_due','invoice_currency_avg']
        print(data[features])
        # data should be a dataFrame and not a numpy array
        predictions = model.predict(data[features])
        data['predicted_amount'] = predictions
        
        data['diff'] =  temp['invoice_amount_doc_currency'] - data['predicted_amount']
        data['predicted_amount'] = data['predicted_amount'].map(lambda x: -x if x < 0 else x)
        data['diff'] = data['diff'].map(lambda x: 0 if x < 0 else x)
        data['predicted_payment_type'] = data['diff'].map(lambda x: "Fully Paid" if x == 0 else "Partially Paid")
        data['actual_open_amount'] = temp['actual_open_amount']
        data['document_id'] = temp['document_id']
        
        #pred = data.loc[:,['actual_open_amount','predictions']].to_dict(orient="records")
        pred = data.to_dict(orient="records")
        return pred
