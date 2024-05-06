from flask import (Flask, render_template, jsonify, request)
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import pyarrow
import io
import requests
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

# get knn model & datasets from local file
try:
    with open('knn_model.pickle', 'rb') as f:
        knn_model = pickle.load(f)
except FileNotFoundError:
    raise Exception("Could not find knn_model.pickle. Please make sure it is in the same folder as the app.py file.")
except Exception as e:
    raise Exception(f"Error loading knn_model.pickle: {str(e)}")

try:
    ratings_pivot = pd.read_parquet('ratings_pivot.parquet')
except FileNotFoundError:
    raise Exception("Could not find ratings_pivot.parquet. Please make sure it is in the same folder as the app.py file.")
except Exception as e:
    raise Exception(f"Error loading ratings_pivot.parquet: {str(e)}")

try:
    filtered_df = pd.read_csv('filtered_df.csv')
except FileNotFoundError:
    raise Exception("Could not find filtered_df.csv. Please make sure it is in the same folder as the app.py file.")
except Exception as e:
    raise Exception(f"Error loading filtered_df.csv: {str(e)}")

def make_recommendations(user, neighbors=11, books=10, model=knn_model):
    distances, indices = model.kneighbors(user.values.reshape(1, -1), n_neighbors = neighbors)

    similar_users = indices.flatten()[1:]
    aggregated_ratings = np.zeros(ratings_pivot.shape[1]) 
    for neighbor_index, distance in zip(similar_users, distances.flatten()[1:]):
            neighbor = ratings_pivot.iloc[neighbor_index]
            aggregated_ratings += neighbor.values * (1 - distance)

    aggregated_df = pd.DataFrame(aggregated_ratings, index=ratings_pivot.columns, columns=['Total_Rating'])
    sorted_books = aggregated_df.sort_values(by='Total_Rating', ascending=False).index.tolist()
    user_rated_books = user[user != 0]
    recommended_books = [book for book in sorted_books if book not in user_rated_books][:books]
    return recommended_books

def recommend_to_names(user):
    recs = make_recommendations(user)
    predictions = {}
    for rec in recs:
        try:
            predictions[rec] = filtered_df[filtered_df['ISBN'] == rec].iloc[0]["Book-Title"]
        except:
            predictions[rec] = rec
    return predictions

@app.route('/')
@app.route("/<name>")
def process(name=None):
    return render_template('hello.html', name=name)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        input_data = request.json
        
        # build user vector
        book_isbns = ratings_pivot.columns
        row_series = pd.Series(0, index=book_isbns)
        for key, value in input_data.items():
            isbn = key
            rating = int(value)
            row_series.loc[isbn] = rating
        # make predictions
        predictions = recommend_to_names(user=row_series)
        return jsonify(predictions), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run()