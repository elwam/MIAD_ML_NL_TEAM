import re
import pickle
import pandas as pd
from flask import Flask, jsonify
from flask_restx import Api, Resource, fields
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from flask_cors import CORS

# Cargar los modelos y objetos necesarios
model = pickle.load(open('modelo_clasificacion_entrenado.pkl', 'rb'))
vect = pickle.load(open('vect.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_transformer.pkl', 'rb'))

# Géneros disponibles
cols = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
        'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
        'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

# Función de expansión de contracciones
def decontracted(phrase):
    # Específicas
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", "can not", phrase)

    # Generales
    phrase = re.sub(r"n't", " not", phrase)
    phrase = re.sub(r"'re", " are", phrase)
    phrase = re.sub(r"'s", " is", phrase)
    phrase = re.sub(r"'d", " would", phrase)
    phrase = re.sub(r"'ll", " will", phrase)
    phrase = re.sub(r"'t", " not", phrase)
    phrase = re.sub(r"'ve", " have", phrase)
    phrase = re.sub(r"'m", " am", phrase)
    return phrase

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

# Crear la aplicación Flask
app = Flask(__name__)
CORS(app) 
#api = Api(app)

api = Api(
    app, 
    version='1.0', 
    title='Clasificación de generos de películas',
    description='Prediction API')


# Definir el namespace y el parser para los endpoints
namespace = api.namespace('predict', description='Clasificación de películas')

parser = api.parser()
parser.add_argument('plot', type=str, required=True, help='Trama de la película', location='args')

resource_fields = api.model('Resource', {
    'genres': fields.List(fields.String),
})

@namespace.route('/')
class ClasificationApi(Resource):

    @api.doc(parser=parser)
    def get(self):
        # Obtener los parámetros de entrada
        args = parser.parse_args()
        plot = args['plot']
        
        
        # Preprocesar los datos de entrada
        preprocessed_plot = decontracted(plot)
        preprocessed_plot = re.sub('[^A-Za-z]+', ' ', preprocessed_plot)
        preprocessed_plot = ' '.join(e.lower() for e in preprocessed_plot.split() if e.lower() not in stopwords)
        
        # Crear el DataFrame con los datos preprocesados
        df = pd.DataFrame({'Plot': [preprocessed_plot]})
        
        # Transformar los datos de entrada utilizando el vectorizador y el transformador TF-IDF
        X_test_dtm_1 = vect.transform(df['Plot'])
        X_test_dtm = tfidf.transform(X_test_dtm_1)
        
        # Realizar la predicción utilizando el modelo entrenado
        predicted_genres = model.predict_proba(X_test_dtm)
        
        # Obtener los géneros basados en las columnas proporcionadas
        movie_genres = [col for pred, col in zip(predicted_genres[0], cols) if pred >= 0.25]
                
        if not movie_genres:
            movie_genres = ["Sin Clasificación"]
        
        # Devolver los géneros como una lista
        return jsonify({'Genres': movie_genres})


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
