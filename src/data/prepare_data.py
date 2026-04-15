import pandas as pd
import numpy as np
import re
import hashlib

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer


class PrepareData:
    def __init__(self, n_artist_features=10, n_name_features=10):
        self.n_artist_features = n_artist_features
        self.n_name_features = n_name_features

        self.vectorizer = HashingVectorizer(
            n_features=self.n_name_features,
            lowercase=True,
            ngram_range=(1, 3),
            alternate_sign=False
        )

    def clean_string(self, text: str) -> str:
        if pd.isna(text) or text == 'unknown':
            return 'unknown'

        text = re.sub(r'[\(\[].*?[\)\]]', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())

        return text.lower()

    def hash_artist(self, artist: str) -> int:
        h = int(hashlib.md5(artist.encode('utf-8')).hexdigest(), 16)
        return h % self.n_artist_features

    def encode_artists(self, artists_column):
        X = np.zeros((len(artists_column), self.n_artist_features))

        for i, artists in enumerate(artists_column):
            if isinstance(artists, str):
                try:
                    artists = eval(artists)
                except:
                    artists = ['unknown']

            if not artists:
                artists = ['unknown']

            cleaned = [self.clean_string(a) for a in artists]

            for artist in cleaned:
                idx = self.hash_artist(artist)
                X[i, idx] += 1

            X[i] /= len(cleaned)

        return X

    def encode_names(self, names_column):
        cleaned = names_column.fillna('unknown').apply(self.clean_string)
        X = self.vectorizer.transform(cleaned)
        return X.toarray()

    def preprocess(self, df: pd.DataFrame):
        df = df.copy()

        if 'popularity' in df.columns:
            y = df['popularity'].values
            df = df.drop(columns=['popularity'])
        else:
            y = None

        df = df.drop(columns=[c for c in ['id', 'release_date'] if c in df.columns])

        numeric_cols = [
            'valence', 'acousticness', 'danceability', 'energy',
            'instrumentalness', 'liveness', 'speechiness',
            'loudness', 'duration_ms', 'tempo', 'year'
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        if 'explicit' in df.columns:
                df['explicit'] = df['explicit'].fillna(0)
        
        if 'n_artists' in df.columns:
            df['n_artists'] = df['n_artists'].fillna(1)

        for col in ['key', 'mode']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])

        if 'artists' in df.columns:
            df['artists'] = df['artists'].fillna("['unknown']")
            X_artists = self.encode_artists(df['artists'])
        else:
            X_artists = np.zeros((len(df), self.n_artist_features))

        if 'name' in df.columns:
            X_names = self.encode_names(df['name'])
        else:
            X_names = np.zeros((len(df), self.n_name_features))

        df = df.drop(columns=['artists', 'name'], errors='ignore')

        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            print(f"Columns with remaining NaN: {nan_cols}")
        else:
            print("No columns with NaN values remain")


        X_num  = df.values

        return X_num, X_artists, X_names, y



