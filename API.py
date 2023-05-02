from flask import Flask, jsonify
from flask_restx import Api, Resource, fields, reqparse
import pickle
from watchdog.events import EVENT_TYPE_CREATED, EVENT_TYPE_OPENED
import pandas as pd

app = Flask(__name__)
api = Api(app, version='1.0', title='Modelo de Predicción de Precios de Autos',
          description='API para predecir el precio de un auto según sus características TEAM WPS')


# Procesamiento de los datos de Entrada
#Importación del modelo a utilizar 
model = pickle.load(open('modelo_entrenado.pkl', 'rb'))

pipeline = pickle.load(open('preprocessor.pkl', 'rb'))

# Creación del namespace para los endpoints
namespace = api.namespace('predict', description='Predicción de precios de vehiculos')

parser = api.parser()
parser.add_argument('Year', type=int, required=True, help='Modelo del Auto (Año)', location='args')
parser.add_argument('Mileage', type=int, required=True, help='Millas Recorridas', location='args')
parser.add_argument('State', type=str, required=True, help='Estado donde se encuentra el vehiculo', location='args')
parser.add_argument('Make', type=str, required=True, help='Marca', location='args')
parser.add_argument('Model', type=str, required=True, help='Modelo', location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})


@api.route('/predict')
class Predict(Resource):
    @api.doc(parser=parser)
    def get(self):
        # Parseamos los parámetros de entrada
        args = parser.parse_args()
        print(f'Argumentos recibidos: {args}')
        
        # Creamos una matriz con los valores de entrada
        input_data = [[args['Year'], args['Mileage'], args['State'], args['Make'], args['Model']]]
        df = pd.DataFrame(input_data, columns=['Year', 'Mileage', 'State', 'Make', 'Model'])
        print(f'DataFrame generado: \n{df}')
        
        # Transformamos los valores de entrada utilizando el preprocesador
        input_data_transformed = pipeline.transform(df)
        print(f'DataFrame transformado: \n{input_data_transformed}')
        
        
        # Realizamos la predicción utilizando el modelo entrenado
        predicted_price = model.predict(input_data_transformed)
        print(f'Precio predicho: {predicted_price[0]}')
        
        
        # Devolvemos la predicción como un diccionario JSON
        return jsonify({'predicted_price': predicted_price[0]})
        
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)