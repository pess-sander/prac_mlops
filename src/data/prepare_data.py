import pandas as pd
import numpy as np
import re
import hashlib

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

        self.scale_ranges = {
            'year': (1900, 2026),
            'duration_ms': (0, 20 * 60 * 1000),
            'loudness': (-60.0, 0.0),
            'tempo': (0.0, 300.0),
            'n_artists': (1.0, 20)
        }

        self.numeric_cols = [
            'valence', 'acousticness', 'danceability', 'energy',
            'instrumentalness', 'liveness', 'speechiness',
            'loudness', 'duration_ms', 'tempo', 'year'
        ]

        self.key_values = list(range(12))
        self.key_columns = [f"key_{k}" for k in self.key_values]

    def clean_string(self, text: str) -> str:
        if pd.isna(text) or text == 'unknown':
            return 'unknown'

        text = re.sub(r'[\(\[].*?[\)\]]', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        return text.lower()
    
    def _clip_other_numeric_to_unit_range(self, df: pd.DataFrame) -> pd.DataFrame:
        scale_cols = set(self.scale_ranges.keys())

        for col in df.columns:
            if col in scale_cols:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
                df[col] = df[col].clip(0, 1)

        return df


    def hash_artist(self, artist: str) -> int:
        h = int(hashlib.md5(artist.encode('utf-8')).hexdigest(), 16)
        return h % self.n_artist_features

    def encode_artists(self, artists_column):
        X = np.zeros((len(artists_column), self.n_artist_features))

        for i, artists in enumerate(artists_column):
            artists = self._parse_artists_value(artists)
            cleaned = [self.clean_string(a) for a in artists]

            for artist in cleaned:
                idx = self.hash_artist(artist)
                X[i, idx] += 1

            X[i] /= len(cleaned)

        return X

    def _parse_artists_value(self, artists):
        if isinstance(artists, str):
            try:
                artists = eval(artists)
            except:
                artists = ['unknown']

        if not artists:
            artists = ['unknown']

        return artists

    def encode_names(self, names_column):
        cleaned = names_column.fillna('unknown').apply(self.clean_string)
        X = self.vectorizer.transform(cleaned)
        return X.toarray()

    def encode_key_one_hot(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'key' not in df.columns:
            return df

        df = df.copy()
        df['key'] = pd.to_numeric(df['key'], errors='coerce')
        df['key'] = df['key'].fillna(-1).astype(int)
        df['key'] = df['key'].clip(-1, 12)

        key_oh = pd.get_dummies(df['key'], prefix='key')
        key_oh = self._align_key_columns(key_oh)

        df = df.drop(columns=['key'])
        df = pd.concat([df, key_oh], axis=1)
        return df

    def _align_key_columns(self, key_oh: pd.DataFrame) -> pd.DataFrame:
        for col in self.key_columns:
            if col not in key_oh.columns:
                key_oh[col] = 0
        return key_oh[self.key_columns]

    def _fill_numeric_missing(self, df: pd.DataFrame, numeric_cols):
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        return df

    def _fill_basic_missing(self, df: pd.DataFrame):
        if 'explicit' in df.columns:
            df['explicit'] = df['explicit'].fillna(0)

        if 'n_artists' in df.columns:
            df['n_artists'] = df['n_artists'].fillna(1)

        for col in ['key', 'mode']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def _process_text_columns(self, df: pd.DataFrame):
        if 'artists' in df.columns:
            df['artists'] = df['artists'].fillna("['unknown']")
            X_artists = self.encode_artists(df['artists'])
        else:
            X_artists = np.zeros((len(df), self.n_artist_features))

        if 'name' in df.columns:
            X_names = self.encode_names(df['name'])
        else:
            X_names = np.zeros((len(df), self.n_name_features))

        return df, X_artists, X_names

    def _scale_selected_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (mn, mx) in self.scale_ranges.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
                df[col] = df[col].clip(mn, mx)
                df[col] = (df[col] - mn) / (mx - mn)
        return df

    def _report_remaining_nans(self, df: pd.DataFrame):
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            print(f"Columns with remaining NaN: {nan_cols}")
        else:
            print("No columns with NaN values remain")

    def preprocess(self, df: pd.DataFrame):
        df = df.copy()

        if 'popularity' in df.columns:
            y = df['popularity'].values
            df = df.drop(columns=['popularity'])
        else:
            y = None

        df = df.drop(columns=[c for c in ['id', 'release_date'] if c in df.columns])

        df = self._fill_numeric_missing(df, self.numeric_cols)
        df = self._fill_basic_missing(df)

        if 'key' in df.columns:
            df = self.encode_key_one_hot(df)

        df, X_artists, X_names = self._process_text_columns(df)
        df = df.drop(columns=['artists', 'name'], errors='ignore')

        df = self._scale_selected_numeric(df)
        df = self._clip_other_numeric_to_unit_range(df)
        self._report_remaining_nans(df)


        X_num = df.values
        return X_num, X_artists, X_names, y



