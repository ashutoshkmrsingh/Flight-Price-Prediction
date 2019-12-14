import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

dataframe = pd.read_excel('Final_Processed_dataset2.xlsx')
dataframe.drop(columns=['Price'], inplace=True)

dataframe.rename(columns=({'Unnamed: 0':'False'}), inplace=True)
dataframe.drop(columns='False', inplace=True)

df = dataframe.copy(deep = True)

name_encode = {'Air Asia': 0, 'Air India': 1, 'GoAir': 2, \
               'IndiGo': 3, 'Jet Airways': 4, 'Jet Airways Business': 5, \
               'Multiple carriers': 6, 'Multiple carriers Premium economy': 7, \
               'SpiceJet': 8, 'Trujet': 9, 'Vistara': 10, 'Vistara Premium economy': 11}

from_encode = {'Banglore': 0, 'Chennai': 1, 'Delhi': 2, 'Kolkata': 3, 'Mumbai': 4}

to_encode = {'Banglore': 0, 'Cochin': 1, 'Delhi': 2, 'Kolkata': 4, 'Hyderabad': 3, 'NewDelhi':5}

extra_services_encode = {
    '1 Long layover' : 0,
 '1 Short layover' : 1,
 '2 Long layover' : 2,
 'Business class' : 3,
 'Change airports': 4,
 'In-flight meal not included' : 5,
 'No Info' : 6,
 'No check-in baggage included' : 7,
 'No info' : 8,
 'Red-eye flight': 9
}


class FlightInterface:
    def __init__(self, input_):
        self.input_ = input_
        self.test = {
            'Name' : None,
            'From' : None,
            'To' : None,
            'Time of Flight' : None,
            'Stops' : None,
            'Extra Services' : None,
            'Price' : None,
            'Day' : None,
            'Month' : None,
            'Route_1' : 0,
            'Route_2' : 0,
            'Route_3' : 0,
            'Route_4' : 0,
            'Route_5' : 0,
            'Dep_hour' : None,
            'Arrival_hour' : None
        }
        
    def encode_input(self):
        self.input_[0] = name_encode[self.input_[0]] # encoding name
        self.input_[1] = from_encode[self.input_[1]] # encoding from
        self.input_[2] = to_encode[self.input_[2]] # encoding to
        self.input_[3] = extra_services_encode[self.input_[3]] # encoding extra services 
        
    def prepare_input(self):
        d = df.loc[(df['Name'] == self.input_[0]) & (df['From'] == self.input_[1]) & (df['To'] == self.input_[2]) & (df.Day == self.input_[4])].sort_values(by='Time of Flight')
        assert d.values.size != 0, 'Flight Not Found!'

        self.test['Name'] = d.iloc[0].Name
        self.test['From'] = d.iloc[0].From
        self.test['To'] = d.iloc[0].To
        self.test['Time of Flight'] = d.iloc[0]['Time of Flight']
        self.test['Stops'] = d.iloc[0].Stops
        self.test['Extra Services'] = self.input_[3]
        # self.test['Extra Services'] = d.iloc[0]['Extra Services']
        self.test['Day'] = d.iloc[0]['Day']
        self.test['Month'] = self.input_[5]
        self.test['Route_1'] = d.iloc[0].Route_1
        self.test['Route_2'] = d.iloc[0].Route_2
        self.test['Route_3'] = d.iloc[0].Route_3
        self.test['Route_4'] = d.iloc[0].Route_4
        self.test['Route_5'] = d.iloc[0].Route_5
        self.test['Dep_hour'] = d.iloc[0].Dep_hour
        self.test['Arrival_hour'] = d.iloc[0].Arrival_hour
    
    def scale_input(self):
        global df
        df2 = df.append(self.test, ignore_index=True)
        df2.drop(columns=['Price'], inplace=True)
        scaler = MinMaxScaler(feature_range=(0,1000))
        scaler.fit(df2)
        df2 = scaler.transform(df2)
        self.scaled_data = df2[-1]
        
    def predict(self):
        model = joblib.load('xgb2.pkl')
        self.test['Price'] = model.predict([self.scaled_data])
        return self.test
    
df = dataframe.copy(deep = True)