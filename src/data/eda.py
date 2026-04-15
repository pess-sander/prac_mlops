import os
import matplotlib.pyplot as plt


class EDAReporter:
    def __init__(self, config):
        self.dir = config["analysis"]["reports_dir"]
        self.key_map = {
            0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
            6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"
        }
        plt.style.use("ggplot")
        os.makedirs(self.dir, exist_ok=True)

    def format_key_mode(self, key, mode):
        if key is None or mode is None:
            return None

        try:
            key_name = self.key_map.get(int(key))
            if key_name is None:
                return None

            return f"{key_name}Maj" if int(mode) == 1 else f"{key_name}m"
        except:
            return None

    def generate(self, batch, batch_id):
        paths = []
        batch = batch.copy()
        batch["key_mode"] = batch.apply(
            lambda row: self.format_key_mode(row["key"], row["mode"]),
            axis=1
        )

        # numeric histograms
        skip_cols = ['artists', 'id', 'name', 'release_date', 'mode', 'key']
        categorical_columns = ['year', 'explicit', 'key_mode']
        num_cols = [x for x in batch.select_dtypes(include="number").columns if x not in categorical_columns and x not in skip_cols]

        for col in num_cols:
            plt.figure(figsize=(8, 6))
            batch[col].dropna().hist(bins=30, color="#4C72B0")
            plt.title(f"{col} distribution (batch {batch_id})", fontsize=12)

            path = os.path.join(self.dir, f"{batch_id}_{col}_hist.png")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()

            paths.append(path)

        cat_cols = [x for x in batch.select_dtypes(include="object").columns if x not in skip_cols]
        cat_cols.extend(categorical_columns)

        for col in cat_cols:
            vc = batch[col].value_counts()

            if len(vc) == 0:
                continue

            plt.figure(figsize=(8, 6))
            vc.plot(kind="bar", color="#4C72B0")
            plt.title(f"{col} top values (batch {batch_id})", fontsize=12)

            path = os.path.join(self.dir, f"{batch_id}_{col}_bar.png")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()

            paths.append(path)

        return paths