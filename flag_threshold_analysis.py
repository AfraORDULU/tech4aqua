import numpy as np

from src.preprocessing import load_and_preprocess_data


DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

CHECKS = [
    ("FLAG_CLORO_CRITICO", "Desinfetante residual"),
    ("FLAG_CALOR_EXTREMO", "TEMP MÁX (ºC)"),
    ("FLAG_RUNOFF_RISCO", "PRECIPITAÇÃO (mm)")
]


def main():

    print("=" * 60)
    print("GOREV 24.1.4 - FLAG ESIK ANALIZI")
    print("=" * 60)

    X, y, groups, _ = load_and_preprocess_data(DATA_PATH)

    for flag, source in CHECKS:

        df = X[[flag, source]].dropna().copy()

        print("\n" + "=" * 60)
        print("FLAG:", flag)
        print("KAYNAK:", source)
        print("=" * 60)

        for flag_value in [0, 1]:

            subset = df.loc[df[flag] == flag_value, source]

            print(f"\nFLAG = {flag_value}")
            print("Gozlem sayisi:", len(subset))

            if not subset.empty:
                print("Minimum:", subset.min())
                print("Maksimum:", subset.max())
                print("Ortalama:", round(subset.mean(), 4))

        values = np.sort(df[source].unique())

        if len(values) < 2:
            print("\nEsik aramasi icin yeterli deger yok.")
            continue

        thresholds = (
            values[:-1] + values[1:]
        ) / 2

        print("\nEN IYI TEK DEGISKENLI ESIKLER")

        for direction in ["less", "greater"]:

            best_accuracy = -1
            best_threshold = None

            for threshold in thresholds:

                if direction == "less":
                    predicted = (
                        df[source] < threshold
                    ).astype(int)
                else:
                    predicted = (
                        df[source] > threshold
                    ).astype(int)

                accuracy = (
                    predicted == df[flag]
                ).mean()

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold

            print(
                f"Yon: {direction} | "
                f"Esik: {best_threshold:.4f} | "
                f"Uyum: %{best_accuracy * 100:.2f}"
            )

    print("\n" + "=" * 60)
    print("GOREV 24.1.4 TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    main()