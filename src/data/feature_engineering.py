class FeatureEngineer:
    def transform(self, batch):
        df = batch.copy()

        if "artists" in df.columns:
            parsed = df["artists"].apply(self.parse_artists)

            df["n_artists"] = parsed.apply(len)

        return df

    def parse_artists(self, s):
        if s is None:
            return []

        parts = str(s).strip("[]").split(",")
        return [p.strip().strip("'").strip('"') for p in parts if p.strip()]