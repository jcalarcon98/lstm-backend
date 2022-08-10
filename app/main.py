from fastapi import FastAPI
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from app.lstm.lstm import get_countries_prediction

# Middlewares
from app.managers.country import CountryManager
from app.serializers.prediction import Prediction

origins = [
    "http://localhost:3000",
]

middleware = [
    Middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
]

app = FastAPI(middleware=middleware)


@app.get('/countries')
def get_countries():
    countries = CountryManager.get_countries()
    return {'countries': countries}


@app.post('/predictions')
def prediction(prediction: Prediction):
    countries = []
    for country in prediction.countries:
        countries.append({'name': country, 'days': prediction.days})
    return get_countries_prediction(countries)
