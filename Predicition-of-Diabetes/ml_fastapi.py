## import libraries
## define base model (should have all the variables which required as input fro predictions)
## load the model
## create a post comand
## post command is when the user sends the data to our api to get the predictions

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

class model_input(BaseModel):
    pregnancies : int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI:float
    DiabetesPedigreeFunction: float
    Age:int

# load the saved model
diabetes_model = pickle.load(open('diabetes_model.sav','rb'))

@app.post('/diabetes_prediction') #/diabetes_prediction is endpoint
def diabetes_pred(input_parameters:model_input):
    ''' ONce we deploy the API users will send data in json format
    we have to convert the data into dictionary. 

    .json() == parses json content from repsonse object and converts it into 
    python dictionary or list

    .loads() == takes JSON formatted string as input and returns a
    Python object. 
    Used when JSON data is available as a string,
    for example when reading a file , receiving data from a web API 
    '''
    input_data = input_parameters.json() # data is posted to api in json format
    input_dictionary = json.loads(input_data) # load jsons

    pregs = input_dictionary['pregnancies']
    glu = input_dictionary['Glucose']
    Bp = input_dictionary['BloodPressure']
    Skinthick = input_dictionary['SkinThickness']
    Insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    DPF = input_dictionary['DiabetesPedigreeFunction'] 
    Age = input_dictionary['Age']

    input_list = [pregs,glu,Bp,Skinthick,Insulin,bmi,DPF,Age]

    prediction = diabetes_model.predict([input_list]) ## put it in a list to tell model predict for one data point

    if prediction[0] == 0:
        return 'The Person is non diabetic'
    else:
        return 'The person is diabetic'
