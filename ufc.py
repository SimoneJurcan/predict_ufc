from pathlib import Path
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import itertools

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("ufc-master.csv")
MODEL_DATASET_PATH = Path("ufc_model_dataset.csv")
RESULTS_PATH = Path("results/model_results.csv")
FIGURES_DIR = Path("results/figures")
SPLIT_DATE = pd.Timestamp("2023-01-01")
RANDOM_STATE = 42
DEFAULT_PREDICTION_MODEL_PATH = Path("random_forest_augmented_without_odds.pkl")

FEATURE_COLS = [
    "lose_streak_dif",
    "win_streak_dif",
    "longest_win_streak_dif",
    "win_dif",
    "loss_dif",
    "total_round_dif",
    "total_title_bout_dif",
    "ko_dif",
    "sub_dif",
    "height_dif",
    "reach_dif",
    "age_dif",
    "sig_str_dif",
    "avg_sub_att_dif",
    "avg_td_dif",
]

# Eksperiment B: samo moneyline odds (bez EV koji je deterministička transformacija)
MONEYLINE_ODDS_COLS = ["R_odds", "B_odds"]

# Eksperiment C: odds + EV (puna verzija kao u originalnoj skripti)
ODDS_EV_COLS = ["R_odds", "B_odds", "R_ev", "B_ev"]

METADATA_COLS = [
    "R_fighter",
    "B_fighter",
    "date",
    "weight_class",
    "Winner",
]

PROFILE_COL_PAIRS = {
    "lose_streak_dif": ("R_current_lose_streak", "B_current_lose_streak"),
    "win_streak_dif": ("R_current_win_streak", "B_current_win_streak"),
    "longest_win_streak_dif": ("R_longest_win_streak", "B_longest_win_streak"),
    "win_dif": ("R_wins", "B_wins"),
    "loss_dif": ("R_losses", "B_losses"),
    "total_round_dif": ("R_total_rounds_fought", "B_total_rounds_fought"),
    "total_title_bout_dif": ("R_total_title_bouts", "B_total_title_bouts"),
    "ko_dif": ("R_win_by_KO/TKO", "B_win_by_KO/TKO"),
    "sub_dif": ("R_win_by_Submission", "B_win_by_Submission"),
    "height_dif": ("R_Height_cms", "B_Height_cms"),
    "reach_dif": ("R_Reach_cms", "B_Reach_cms"),
    "age_dif": ("R_age", "B_age"),
    "sig_str_dif": ("R_avg_SIG_STR_landed", "B_avg_SIG_STR_landed"),
    "avg_sub_att_dif": ("R_avg_SUB_ATT", "B_avg_SUB_ATT"),
    "avg_td_dif": ("R_avg_TD_landed", "B_avg_TD_landed"),
}


# ---------------------------------------------------------------------------
# Učitavanje i priprema podataka
# ---------------------------------------------------------------------------

def check_diff_direction(raw_df: pd.DataFrame) -> None:
    """
    Provjeri jesu li dif stupci definirani kao Red - Blue ili Blue - Red.

    U ovom datasetu ocekujemo Blue - Red. Ta informacija je vazna jer kod
    hipotetskih borbi i corner augmentacije moramo znati treba li featuree
    okrenuti predznakom.
    """
    checks = {
        "age_dif": ("R_age", "B_age"),
        "height_dif": ("R_Height_cms", "B_Height_cms"),
        "reach_dif": ("R_Reach_cms", "B_Reach_cms"),
        "win_dif": ("R_wins", "B_wins"),
        "loss_dif": ("R_losses", "B_losses"),
        "sig_str_dif": ("R_avg_SIG_STR_landed", "B_avg_SIG_STR_landed"),
        "avg_td_dif": ("R_avg_TD_landed", "B_avg_TD_landed"),
        "avg_sub_att_dif": ("R_avg_SUB_ATT", "B_avg_SUB_ATT"),
    }

    print("\n" + "=" * 70)
    print("PROVJERA SMJERA DIF STUPACA")
    print("=" * 70)

    blue_minus_red_matches = []

    for diff_col, (r_col, b_col) in checks.items():
        temp = raw_df[[diff_col, r_col, b_col]].dropna()
        if temp.empty:
            continue

        red_minus_blue = (temp[r_col] - temp[b_col]).round(6)
        blue_minus_red = (temp[b_col] - temp[r_col]).round(6)
        actual = temp[diff_col].round(6)

        match_r_minus_b = (actual == red_minus_blue).mean()
        match_b_minus_r = (actual == blue_minus_red).mean()
        blue_minus_red_matches.append(match_b_minus_r)

        print(
            f"{diff_col}: "
            f"R-B match={match_r_minus_b:.3f}, "
            f"B-R match={match_b_minus_r:.3f}"
        )

    if blue_minus_red_matches and min(blue_minus_red_matches) < 0.95:
        print(
            "Napomena: neki dif stupci nisu konzistentni kroz cijeli dataset "
            "pri direktnoj rekonstrukciji iz R/B raw stupaca. To se tretira "
            "kao dijagnostika, a ne kao prekid rada."
        )
    else:
        print("Zakljucak: dif stupci su konzistentno definirani kao Blue - Red.")

    if "date" in raw_df.columns:
        recent_df = raw_df.copy()
        recent_df["date"] = pd.to_datetime(recent_df["date"], errors="coerce")
        recent_df = recent_df.loc[recent_df["date"] >= SPLIT_DATE]
        if not recent_df.empty:
            print("\nProvjera samo za noviji/test dio dataseta:")
            for diff_col, (r_col, b_col) in checks.items():
                temp = recent_df[[diff_col, r_col, b_col]].dropna()
                if temp.empty:
                    continue

                red_minus_blue = (temp[r_col] - temp[b_col]).round(6)
                blue_minus_red = (temp[b_col] - temp[r_col]).round(6)
                actual = temp[diff_col].round(6)
                match_r_minus_b = (actual == red_minus_blue).mean()
                match_b_minus_r = (actual == blue_minus_red).mean()
                print(
                    f"{diff_col}: "
                    f"R-B match={match_r_minus_b:.3f}, "
                    f"B-R match={match_b_minus_r:.3f}"
                )

def check_asof_consistency(raw_df: pd.DataFrame) -> None:
    """
    Provjera da su karijerne statistike 'as-of' (izracunate PRIJE borbe):
    broj pobjeda prije borbe t+1 mora biti broj pobjeda prije borbe t
    uvecan za ishod borbe t. Odstupanja su ocekivana u malom postotku
    (borbe izvan skupa, istoimeni borci) i sluze kao dijagnostika.
    """
    frames = []
    for side, win_label in (("R", "Red"), ("B", "Blue")):
        sub = raw_df[[f"{side}_fighter", "date", f"{side}_wins", "Winner"]].copy()
        sub.columns = ["fighter", "date", "wins_before", "Winner"]
        sub = sub.loc[sub["Winner"].isin(["Red", "Blue"])]
        sub["won"] = (sub["Winner"] == win_label).astype(int)
        frames.append(sub)

    timeline = pd.concat(frames)
    timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
    timeline = timeline.dropna(subset=["date"]).sort_values(["fighter", "date"])

    checked = violations = 0
    for _, group in timeline.groupby("fighter"):
        wins = group["wins_before"].to_numpy()
        won = group["won"].to_numpy()
        for i in range(len(group) - 1):
            checked += 1
            if wins[i + 1] != wins[i] + won[i]:
                violations += 1

    if checked:
        share = 1 - violations / checked
        print(
            f"\nProvjera as-of semantike: {share:.1%} tranzicija konzistentno "
            f"({violations}/{checked} odstupanja)."
        )

def load_prediction_model(model_path: Path) -> Pipeline:
    """Ucitaj model uz provjeru kompatibilnosti; podrzava i stari format."""
    obj = joblib.load(model_path)
    if not (isinstance(obj, dict) and "model" in obj):
        print("Napomena: model je u starom formatu bez metapodataka.")
        return obj
    meta = obj.get("meta", {})
    if meta.get("feature_cols") and meta["feature_cols"] != FEATURE_COLS:
        raise ValueError(
            f"Model {model_path} treniran je sa znacajkama {meta['feature_cols']}, "
            f"a predikcija hipotetskih borbi gradi samo {FEATURE_COLS}. "
            "Koristi model bez odds znacajki (npr. *_augmented_without_odds.pkl)."
        )
    if meta.get("sklearn_version") not in (None, sklearn.__version__):
        print(
            f"UPOZORENJE: model treniran sa scikit-learn {meta['sklearn_version']}, "
            f"trenutno je instaliran {sklearn.__version__}."
        )
    return obj["model"]


def reconstruct_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ponovno izracunaj svih 15 dif znacajki iz sirovih R_/B_ stupaca kao Blue - Red.

    Razlog: u izvornom datasetu neki gotovi dif stupci (npr. age_dif, loss_dif)
    nisu konzistentni sa sirovim R_/B_ vrijednostima (vjerojatno naknadno
    osvjezeni). Buduci da hipotetske borbe grade featuree kao blue_profile -
    red_profile iz istih sirovih stupaca, treniramo na identicnoj, provjerljivoj
    definiciji. Time se uklanja train/serve neslaganje.
    """
    df = df.copy()
    for feature_name, (red_col, blue_col) in PROFILE_COL_PAIRS.items():
        df[feature_name] = df[blue_col] - df[red_col]
    return df


def augment_train_with_swapped_corners(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Za svaki training redak dodaj swapped verziju borbe.

    Buduci da su dif znacajke Blue - Red, zamjena kutova znaci:
      swapped_feature = -original_feature
      swapped_target = 1 - original_target

    Test skup se ne augmentira. Evaluacija ostaje na stvarnim povijesnim
    zapisima, a augmentacija se koristi samo za trening.
    """
    original = train_df.copy()
    swapped = train_df.copy()
    swapped[feature_cols] = -swapped[feature_cols]
    swapped["target"] = 1 - swapped["target"]

    augmented = pd.concat([original, swapped], ignore_index=True)
    return augmented.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    """Učitaj podatke, zadrži završene Red/Blue borbe i napravi target."""
    if not path.exists():
        raise FileNotFoundError(f"Datoteka nije pronađena: {path.resolve()}")

    df = pd.read_csv(path)

    raw_profile_cols = {col for pair in PROFILE_COL_PAIRS.values() for col in pair}
    required_cols = set(METADATA_COLS + ODDS_EV_COLS) | raw_profile_cols
    missing_cols = sorted(required_cols.difference(df.columns))
    if missing_cols:
        raise ValueError(f"U datasetu nedostaju stupci: {missing_cols}")

    print("=" * 70)
    print("OSNOVNE INFORMACIJE O IZVORNOM DATASETU")
    print("=" * 70)
    print(f"Broj redaka: {df.shape[0]}")
    print(f"Broj stupaca: {df.shape[1]}")
    print("\nRaspodjela stupca Winner:")
    print(df["Winner"].value_counts(dropna=False))
    print("\nTipovi podataka:")
    df.info()

    check_diff_direction(df)
    check_asof_consistency(df)

    # Rekonstruiraj dif znacajke iz sirovih R_/B_ stupaca (Blue - Red) kako bi
    # trening koristio identicnu definiciju kao hipotetske predikcije.
    df = reconstruct_diff_features(df)
    print(
        "\nDif znacajke rekonstruirane iz sirovih R_/B_ stupaca kao Blue - Red "
        "(konzistentno s predict_fight)."
    )

    model_df = df.loc[df["Winner"].isin(["Red", "Blue"])].copy()
    model_df["date"] = pd.to_datetime(model_df["date"], errors="coerce")

    invalid_dates = model_df["date"].isna().sum()
    if invalid_dates:
        print(f"\nUklanja se {invalid_dates} redaka s neispravnim datumom.")
        model_df = model_df.dropna(subset=["date"])

    model_df["target"] = (model_df["Winner"] == "Red").astype(int)
    model_df = model_df.sort_values("date").reset_index(drop=True)

    output_cols = METADATA_COLS + FEATURE_COLS + ODDS_EV_COLS + ["target"]
    model_df = model_df[output_cols]
    model_df.to_csv(MODEL_DATASET_PATH, index=False, date_format="%Y-%m-%d")

    print("\n" + "=" * 70)
    print("OČIŠĆENI MODEL DATASET")
    print("=" * 70)
    print(f"Zadržano borbi: {len(model_df)}")
    print(f"Red pobjede (target=1): {(model_df['target'] == 1).sum()}")
    print(f"Blue pobjede (target=0): {(model_df['target'] == 0).sum()}")
    print(f"Spremljeno u: {MODEL_DATASET_PATH.resolve()}")

    return model_df


# ---------------------------------------------------------------------------
# Vremenski split
# ---------------------------------------------------------------------------

def make_time_split(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Podijeli podatke kronološki kako bi test simulirao buduće borbe."""
    train_mask = df["date"] < SPLIT_DATE
    test_mask = df["date"] >= SPLIT_DATE

    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            f"Split na datumu {SPLIT_DATE.date()} mora dati neprazan train i test skup."
        )

    X_train = df.loc[train_mask, feature_cols]
    X_test = df.loc[test_mask, feature_cols]
    y_train = df.loc[train_mask, "target"]
    y_test = df.loc[test_mask, "target"]

    print(f"\nTrain: {len(X_train)} borbi prije {SPLIT_DATE.date()}")
    print(f"Test:  {len(X_test)} borbi od {SPLIT_DATE.date()} nadalje")
    print(
        f"Raspon train skupa: {df.loc[train_mask, 'date'].min().date()} - "
        f"{df.loc[train_mask, 'date'].max().date()}"
    )
    print(
        f"Raspon test skupa:  {df.loc[test_mask, 'date'].min().date()} - "
        f"{df.loc[test_mask, 'date'].max().date()}"
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Izgradnja modela (pipeline-ovi)
# ---------------------------------------------------------------------------

def build_logistic_pipeline(feature_cols: list[str]) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )


def build_random_forest_pipeline(feature_cols: list[str]) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                feature_cols,
            )
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_leaf=20,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_hist_gradient_boosting_pipeline(feature_cols: list[str]) -> Pipeline:
    """
    HistGradientBoostingClassifier nativno podržava NaN vrijednosti,
    pa ne treba imputer. Dobar za tablične podatke bez XGBoost ovisnosti.
    """
    return Pipeline(
        steps=[
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.05,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Evaluacija jednog modela — vraća dict za rezultatnu tablicu
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    experiment_label: str = "",
) -> dict:
    """Trenira model, ispisuje metrike i vraća dict rezultata."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    red_probabilities = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, predictions)
    bal_acc = balanced_accuracy_score(y_test, predictions)
    roc = roc_auc_score(y_test, red_probabilities)
    brier = brier_score_loss(y_test, red_probabilities)

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"ROC-AUC:           {roc:.4f}")
    print(f"Brier score:       {brier:.4f} (nize je bolje, 0.25 = neinformativan)")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Blue (0)", "Red (1)"],
            digits=4,
            zero_division=0,
        )
    )
    print("Confusion matrix (redci=stvarno, stupci=predviđeno):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    return {
        "experiment": experiment_label,
        "model": name,
        "evaluation_scope": "full_test",
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "roc_auc": round(roc, 4),
        "brier_score": round(brier, 4),
        "n_test": len(y_test),
    }


def evaluate_model_on_existing_test_subset(
    name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    subset_mask: pd.Series,
    experiment_label: str,
    evaluation_scope: str,
) -> dict | None:
    """
    Evaluira vec istrenirani model na podskupu testnog skupa.

    Koristi se za fer usporedbu modela s odds znacajkama i odds-favorite
    baselinea: oba se racunaju samo na test borbama gdje postoje R_odds i B_odds.
    Model se ovdje NE trenira ponovno.
    """
    if subset_mask.sum() == 0:
        print(f"\n{name} ({evaluation_scope}): nema redaka za dodatnu evaluaciju.")
        return None

    X_subset = X_test.loc[subset_mask]
    y_subset = y_test.loc[subset_mask]

    predictions = model.predict(X_subset)
    red_probabilities = model.predict_proba(X_subset)[:, 1]

    acc = accuracy_score(y_subset, predictions)
    bal_acc = balanced_accuracy_score(y_subset, predictions)
    roc = roc_auc_score(y_subset, red_probabilities)
    brier = brier_score_loss(y_subset, red_probabilities)

    print("\n" + "-" * 70)
    print(f"{name} - dodatna evaluacija: {evaluation_scope}")
    print("-" * 70)
    print(f"n_test:            {len(y_subset)}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"ROC-AUC:           {roc:.4f}")
    print(f"Brier score:       {brier:.4f}")

    return {
        "experiment": experiment_label,
        "model": name,
        "evaluation_scope": evaluation_scope,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "roc_auc": round(roc, 4),
        "brier_score": round(brier, 4),
        "n_test": len(y_subset),
    }


def print_feature_importance(model: Pipeline, feature_cols: list[str]) -> None:
    forest = model.named_steps["model"]
    importance = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": forest.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("\nRandom Forest feature importance:")
    print(importance.to_string(index=False, formatters={"importance": "{:.4f}".format}))


# ---------------------------------------------------------------------------
# Baseline modeli
# ---------------------------------------------------------------------------

def evaluate_baselines(df: pd.DataFrame, all_results: list[dict]) -> None:
    """
    Izračunaj i ispiši tri baseline modela:
      1. Always-Red   — uvijek predviđa Red pobjedu
      2. Always-Blue  — uvijek predviđa Blue pobjedu
      3. Odds-favorite — predviđa favorita prema moneyline odds-u
                         (samo na redcima gdje oba odds nisu NaN)
    """
    test_mask = df["date"] >= SPLIT_DATE
    test_df = df.loc[test_mask].copy()
    y_test = test_df["target"]

    n_test = len(y_test)
    n_red = (y_test == 1).sum()
    n_blue = (y_test == 0).sum()

    print("\n" + "=" * 70)
    print("BASELINE MODELI")
    print("=" * 70)

    # 1. Always-Red
    always_red_acc = n_red / n_test
    always_red_bal = balanced_accuracy_score(y_test, [1] * n_test)
    print(f"\nAlways-Red baseline   (n={n_test})")
    print(f"  Accuracy:          {always_red_acc:.4f}")
    print(f"  Balanced accuracy: {always_red_bal:.4f}")
    print(f"  ROC-AUC:           N/A (konstantne predikcije)")
    all_results.append({
        "experiment": "Baseline",
        "model": "Always-Red",
        "evaluation_scope": "full_test",
        "accuracy": round(always_red_acc, 4),
        "balanced_accuracy": round(always_red_bal, 4),
        "roc_auc": None,
        "brier_score": None,
        "n_test": n_test,
    })

    # 2. Always-Blue
    always_blue_acc = n_blue / n_test
    always_blue_bal = balanced_accuracy_score(y_test, [0] * n_test)
    print(f"\nAlways-Blue baseline  (n={n_test})")
    print(f"  Accuracy:          {always_blue_acc:.4f}")
    print(f"  Balanced accuracy: {always_blue_bal:.4f}")
    print(f"  ROC-AUC:           N/A (konstantne predikcije)")
    all_results.append({
        "experiment": "Baseline",
        "model": "Always-Blue",
        "evaluation_scope": "full_test",
        "accuracy": round(always_blue_acc, 4),
        "balanced_accuracy": round(always_blue_bal, 4),
        "roc_auc": None,
        "brier_score": None,
        "n_test": n_test,
    })

    # 3. Odds-favorite (samo redci s oba odds dostupna)
    odds_mask = test_df["R_odds"].notna() & test_df["B_odds"].notna()
    odds_df = test_df.loc[odds_mask].copy()
    y_odds = odds_df["target"]
    n_odds = len(y_odds)

    if n_odds == 0:
        print("\nOdds-favorite baseline: nema redaka s odds podacima u test skupu.")
    else:
        # Za ROC-AUC koristimo implicitnu vjerojatnost iz odds-a:
        # Pretvaramo moneyline u implied probability za Red.
        def moneyline_to_implied_prob(odds_series: pd.Series) -> pd.Series:
            odds = odds_series.astype(float)
            if (odds == 0).any():
                raise ValueError("Moneyline koeficijent 0 nije definiran.")
            prob = pd.Series(np.nan, index=odds.index, dtype=float)
            neg_mask = odds < 0
            pos_mask = odds > 0
            prob[neg_mask] = (-odds[neg_mask]) / (-odds[neg_mask] + 100)
            prob[pos_mask] = 100 / (odds[pos_mask] + 100)
            return prob

        r_impl = moneyline_to_implied_prob(odds_df["R_odds"])
        b_impl = moneyline_to_implied_prob(odds_df["B_odds"])
        # Normaliziramo da zbroj bude 1 (uklonimo vig)
        total = r_impl + b_impl
        r_prob_norm = r_impl / total
        # Ako su implied probabilities potpuno jednake, nema jasnog favorita.
        # Strogi ">" ostavlja takve rijetke tie slucajeve kao Blue (0), sto je
        # ekvivalentno ranijoj provjeri R_odds < B_odds i daje usporediv baseline.
        odds_pred = (r_prob_norm > 0.5).astype(int)

        odds_acc = accuracy_score(y_odds, odds_pred)
        odds_bal = balanced_accuracy_score(y_odds, odds_pred)
        odds_roc = roc_auc_score(y_odds, r_prob_norm)
        odds_brier = brier_score_loss(y_odds, r_prob_norm)

        print(f"\nOdds-favorite baseline (n={n_odds}, samo borbe s odds podacima)")
        print(f"  Accuracy:          {odds_acc:.4f}")
        print(f"  Balanced accuracy: {odds_bal:.4f}")
        print(f"  ROC-AUC:           {odds_roc:.4f}")
        print(f"  Brier score:       {odds_brier:.4f}")

        all_results.append({
            "experiment": "Baseline",
            "model": "Odds-Favorite",
            "evaluation_scope": "odds_available_test",
            "accuracy": round(odds_acc, 4),
            "balanced_accuracy": round(odds_bal, 4),
            "roc_auc": round(odds_roc, 4),
            "brier_score": round(odds_brier, 4),
            "n_test": n_odds,
        })


# ---------------------------------------------------------------------------
# Pokretanje jednog eksperimenta (LR + RF + HGB)
# ---------------------------------------------------------------------------

def run_experiment(
    df: pd.DataFrame,
    experiment_name: str,
    feature_cols: list[str],
    file_suffix: str,
    all_results: list[dict],
    augment: bool = False,
) -> None:
    """
    Treniraj LR + RF + HGB za jedan eksperiment i evaluiraj na test skupu.

    Ako je augment=True, train skup se prosiruje zamjenom kutova (za smanjenje
    Red/Blue pristranosti); test skup ostaje originalan u oba slucaja.
    """
    print("\n" + "=" * 70)
    print(experiment_name)
    print("=" * 70)
    print(f"Broj featurea: {len(feature_cols)}")
    print(", ".join(feature_cols))

    train_mask = df["date"] < SPLIT_DATE
    test_mask = df["date"] >= SPLIT_DATE
    odds_available_mask = (
        df.loc[test_mask, "R_odds"].notna() & df.loc[test_mask, "B_odds"].notna()
    )

    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, "target"]

    if augment:
        train_df = df.loc[train_mask, feature_cols + ["target"]].copy()
        augmented_train_df = augment_train_with_swapped_corners(train_df, feature_cols)
        X_train = augmented_train_df[feature_cols]
        y_train = augmented_train_df["target"]
        scope_full = "full_test_augmented_train"
        scope_subset = "odds_available_test_augmented_train"
        print(f"\nOriginal train:  {int(train_mask.sum())} borbi prije {SPLIT_DATE.date()}")
        print(f"Augmented train: {len(augmented_train_df)} redaka nakon zamjene kutova")
        print(f"Test:            {int(test_mask.sum())} borbi od {SPLIT_DATE.date()} nadalje")
    else:
        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, "target"]
        scope_full = "full_test"
        scope_subset = "odds_available_test"
        make_time_split(df, feature_cols)  # ispis raspona train/test skupa

    logistic_model = build_logistic_pipeline(feature_cols)
    forest_model = build_random_forest_pipeline(feature_cols)
    hgb_model = build_hist_gradient_boosting_pipeline(feature_cols)

    for model_obj, model_name, suffix in [
        (logistic_model, "Logistic Regression", "logistic_regression"),
        (forest_model, "Random Forest", "random_forest"),
        (hgb_model, "HistGradientBoosting", "hist_gradient_boosting"),
    ]:
        result = evaluate_model(
            model_name,
            model_obj,
            X_train,
            X_test,
            y_train,
            y_test,
            experiment_label=experiment_name,
        )
        result["evaluation_scope"] = scope_full
        all_results.append(result)

        subset_result = evaluate_model_on_existing_test_subset(
            model_name,
            model_obj,
            X_test,
            y_test,
            odds_available_mask,
            experiment_name,
            scope_subset,
        )
        if subset_result is not None:
            all_results.append(subset_result)

        model_path = Path(f"{suffix}_{file_suffix}.pkl")
        joblib.dump(
            {
                "model": model_obj,
                "meta": {
                    "experiment": experiment_name,
                    "feature_cols": list(feature_cols),
                    "split_date": str(SPLIT_DATE.date()),
                    "augmented_train": augment,
                    "sklearn_version": sklearn.__version__,
                    "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                },
            },
            model_path,
        )

    print_feature_importance(forest_model, feature_cols)


# ---------------------------------------------------------------------------
# Rezultatna tablica
# ---------------------------------------------------------------------------

def save_results_table(all_results: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(all_results)

    # Sortiranje: eksperiment → model
    results_df = results_df.sort_values(
        ["experiment", "evaluation_scope", "model"]
    ).reset_index(drop=True)

    results_df.to_csv(RESULTS_PATH, index=False)

    print("\n" + "=" * 70)
    print("USPOREDNA TABLICA SVIH REZULTATA")
    print("=" * 70)
    print(
        results_df.to_string(
            index=False,
            formatters={
                "accuracy": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
                "balanced_accuracy": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
                "roc_auc": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
                "brier_score": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
            },
        )
    )
    print(f"\nTablica spremljena u: {RESULTS_PATH.resolve()}")


# ---------------------------------------------------------------------------
# Predikcija hipotetske borbe
# ---------------------------------------------------------------------------

def find_latest_fighter_profile(source_df: pd.DataFrame, fighter_name: str, weight_class: str = None, warn: bool = True) -> dict:
    """
    Pronadji zadnji poznati profil borca u originalnom datasetu.

    Borac se moze pojaviti kao Red ili Blue. Zato se vrijednosti uzimaju iz
    R_* stupaca ako je bio Red, odnosno iz B_* stupaca ako je bio Blue.
    """
    required_cols = {"R_fighter", "B_fighter", "date", "weight_class"}
    for red_col, blue_col in PROFILE_COL_PAIRS.values():
        required_cols.add(red_col)
        required_cols.add(blue_col)

    missing_cols = sorted(required_cols.difference(source_df.columns))
    if missing_cols:
        raise ValueError(f"U izvornom datasetu nedostaju stupci: {missing_cols}")

    df = source_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    fighter_key = fighter_name.casefold().strip()
    red_mask = df["R_fighter"].astype(str).str.casefold().str.strip() == fighter_key
    blue_mask = df["B_fighter"].astype(str).str.casefold().str.strip() == fighter_key

    fighter_rows = df.loc[red_mask | blue_mask].dropna(subset=["date"]).copy()
    if fighter_rows.empty:
        available_names = pd.concat([df["R_fighter"], df["B_fighter"]]).dropna().unique()
        suggestions = [
            name
            for name in available_names
            if fighter_key in str(name).casefold() or str(name).casefold() in fighter_key
        ][:10]
        suggestion_text = f" Moguci slicni nazivi: {suggestions}" if suggestions else ""
        raise ValueError(f"Borac nije pronadjen u datasetu: {fighter_name}.{suggestion_text}")


    # NOVO: filtar po kategoriji + upozorenje na moguću koliziju imena

    if weight_class is not None:
        fighter_rows = fighter_rows.loc[fighter_rows["weight_class"] == weight_class]
        if fighter_rows.empty:
            raise ValueError(
                f"Borac {fighter_name} nema borbi u kategoriji {weight_class}."
            )
    else:
        classes = sorted(fighter_rows["weight_class"].dropna().unique())
        if warn and len(classes) > 1:
            print(
                f"UPOZORENJE: ime '{fighter_name}' pojavljuje se u vise kategorija "
                f"({', '.join(classes)}). Moguce je da se radi o razlicitim osobama "
                f"istog imena; profil se uzima iz najnovije borbe. Za jednoznacan "
                f"odabir proslijedi weight_class."
            )

    latest_row = fighter_rows.sort_values("date").iloc[-1]
    side = "R" if str(latest_row["R_fighter"]).casefold().strip() == fighter_key else "B"

    profile = {
        "fighter_name": latest_row[f"{side}_fighter"],
        "latest_date": latest_row["date"],
        "weight_class": latest_row["weight_class"],
        "side_in_latest_fight": "Red" if side == "R" else "Blue",
    }

    for feature_name, (red_col, blue_col) in PROFILE_COL_PAIRS.items():
        source_col = red_col if side == "R" else blue_col
        profile[feature_name] = latest_row[source_col]

    # Rang u tezinskoj kategoriji (0 = prvak, 1-15 = rangirani, NaN = nerangiran).
    # Koristi se samo kao filtar realnih protivnika u recommenderu, ne kao feature.
    rank_col = f"{side}_match_weightclass_rank"
    profile["weightclass_rank"] = (
        latest_row[rank_col] if rank_col in latest_row.index else float("nan")
    )

    return profile


def get_latest_profiles_by_weight_class(source_df: pd.DataFrame) -> pd.DataFrame:
    """
    Izgradi tablicu zadnjih poznatih profila svih boraca.

    Svaki borac se pojavljuje jednom, s podacima iz svoje najnovije borbe u datasetu.
    """
    names = pd.concat([source_df["R_fighter"], source_df["B_fighter"]]).dropna().unique()
    profiles = []

    for name in names:
        try:
            # warn=False: kod masovnog nabrajanja ne zelimo stotine upozorenja
            # o borcima koji su mijenjali kategoriju.
            profile = find_latest_fighter_profile(source_df, str(name), warn=False)
        except ValueError:
            continue
        profiles.append(profile)

    profiles_df = pd.DataFrame(profiles)
    if profiles_df.empty:
        return profiles_df

    return profiles_df.sort_values(
        ["weight_class", "fighter_name"]
    ).reset_index(drop=True)


def build_hypothetical_feature_row(red_profile: dict, blue_profile: dict) -> pd.DataFrame:
    """
    Napravi jedan redak featurea za hipotetsku borbu.

    Vazno: u datasetu su dif stupci definirani kao Blue - Red.
    Zato i ovdje racunamo blue_profile - red_profile.
    """
    feature_values = {}
    for feature_name in FEATURE_COLS:
        feature_values[feature_name] = blue_profile[feature_name] - red_profile[feature_name]

    return pd.DataFrame([feature_values], columns=FEATURE_COLS)


def predict_red_probability(model: Pipeline, red_profile: dict, blue_profile: dict) -> float:
    """Vrati vjerojatnost pobjede Red borca za zadani raspored kutova."""
    fight_features = build_hypothetical_feature_row(red_profile, blue_profile)
    return float(model.predict_proba(fight_features)[0, 1])


def predict_fight(
    fighter_a: str,
    fighter_b: str,
    model_path: Path | str = DEFAULT_PREDICTION_MODEL_PATH,
    enforce_same_weight_class: bool = True,
    neutralize_corner: bool = True,
) -> dict:
    """
    Predvidi ishod hipotetske borbe koristeci osnovne dif znacajke bez odds-a.

    Default je neutralna predikcija: borba se simulira u oba rasporeda kutova
    i vjerojatnosti se prosjece. Tako korisnik ne mora znati tko je Red/Blue.

    Primjer:
        predict_fight("Israel Adesanya", "Brendan Allen")
    """
    source_df = pd.read_csv(DATA_PATH)

    fighter_a_profile = find_latest_fighter_profile(source_df, fighter_a)
    fighter_b_profile = find_latest_fighter_profile(source_df, fighter_b)

    if (
        enforce_same_weight_class
        and fighter_a_profile["weight_class"] != fighter_b_profile["weight_class"]
    ):
        raise ValueError(
            "Borci nemaju istu zadnju poznatu tezinsku kategoriju: "
            f"{fighter_a_profile['fighter_name']} = {fighter_a_profile['weight_class']}, "
            f"{fighter_b_profile['fighter_name']} = {fighter_b_profile['weight_class']}."
        )

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model nije pronadjen: {model_path.resolve()}. "
            "Prvo pokreni: python zavrsni.py --train"
        )

    model = load_prediction_model(model_path)


    fighter_a_as_red_probability = predict_red_probability(
        model,
        fighter_a_profile,
        fighter_b_profile,
    )
    fighter_a_as_blue_probability = 1 - predict_red_probability(
        model,
        fighter_b_profile,
        fighter_a_profile,
    )

    if neutralize_corner:
        fighter_a_probability = (
            fighter_a_as_red_probability + fighter_a_as_blue_probability
        ) / 2
        prediction_mode = "neutralized_corners"
    else:
        fighter_a_probability = fighter_a_as_red_probability
        prediction_mode = "respect_input_order_as_red_blue"

    fighter_b_probability = 1 - fighter_a_probability
    predicted_winner = (
        fighter_a_profile["fighter_name"]
        if fighter_a_probability >= 0.5
        else fighter_b_profile["fighter_name"]
    )

    result = {
        "fight": f"{fighter_a_profile['fighter_name']} vs {fighter_b_profile['fighter_name']}",
        "model": str(model_path),
        "prediction_mode": prediction_mode,
        "predicted_winner": predicted_winner,
        "fighter_a": fighter_a_profile["fighter_name"],
        "fighter_b": fighter_b_profile["fighter_name"],
        "fighter_a_win_probability": round(fighter_a_probability, 4),
        "fighter_b_win_probability": round(fighter_b_probability, 4),
        "fighter_a_as_red_probability": round(fighter_a_as_red_probability, 4),
        "fighter_a_as_blue_probability": round(fighter_a_as_blue_probability, 4),
        "fighter_a_latest_profile_date": fighter_a_profile["latest_date"].date().isoformat(),
        "fighter_b_latest_profile_date": fighter_b_profile["latest_date"].date().isoformat(),
        "weight_class": fighter_a_profile["weight_class"],
    }

    print("\n" + "=" * 70)
    print("PREDIKCIJA HIPOTETSKE BORBE")
    print("=" * 70)
    print(f"Fight: {result['fight']}")
    print(f"Weight class: {result['weight_class']}")
    print(f"Model: {result['model']}")
    print(f"Prediction mode: {result['prediction_mode']}")
    print(f"Predicted winner: {result['predicted_winner']}")
    print(
        f"{result['fighter_a']} win probability: "
        f"{result['fighter_a_win_probability']:.4f}"
    )
    print(
        f"{result['fighter_b']} win probability: "
        f"{result['fighter_b_win_probability']:.4f}"
    )
    if neutralize_corner:
        print(
            "Corner sensitivity: "
            f"{result['fighter_a']} as Red = {result['fighter_a_as_red_probability']:.4f}, "
            f"{result['fighter_a']} as Blue = {result['fighter_a_as_blue_probability']:.4f}"
        )
    today = pd.Timestamp.now().normalize()
    age_a = (today - fighter_a_profile["latest_date"]).days
    age_b = (today - fighter_b_profile["latest_date"]).days
    print(
        "Latest profiles: "
        f"{result['fighter_a']} ({result['fighter_a_latest_profile_date']}, {age_a} dana), "
        f"{result['fighter_b']} ({result['fighter_b_latest_profile_date']}, {age_b} dana)"
    )
    print(
        "Napomena: profil je 'as-of' prije zadnje borbe borca (statistike ne "
        "ukljucuju ishod te borbe ni protok vremena; dob nije azurirana na danas)."
    )

    return result


def format_weightclass_rank(rank: float) -> str:
    """Citljiv prikaz ranga: 0 = prvak, 1-15 = #N, NaN = nerangiran."""
    if pd.isna(rank):
        return "nerangiran"
    if int(rank) == 0:
        return "prvak"
    return f"#{int(rank)}"


def recommend_matchups(
    fighter_name: str,
    top_n: int = 10,
    model_path: Path | str = DEFAULT_PREDICTION_MODEL_PATH,
    active_since: str | pd.Timestamp | None = SPLIT_DATE,
    include_unranked: bool = False,
) -> pd.DataFrame:
    """
    Predlozi najkompetitivnije protivnike u istoj tezinskoj kategoriji.

    Kompetitivnost = blizina neutralizirane vjerojatnosti 50/50:
        competitiveness_score = 1 - 2 * abs(fighter_win_probability - 0.5)
    Veci score (blize 1.0) znaci izjednaceniji matchup.

    Po defaultu se predlazu samo RANGIRANI protivnici (prvak ili top 15 u
    kategoriji), jer model bez odds-a ne razlikuje razinu borca pa bi inace
    predlagao statisticki slicne, ali nerealne protivnike (npr. nerangirane
    zurnejmene). Rang je vanjski/ekspertni signal i koristi se samo za odabir
    realnog skupa protivnika, ne kao znacajka za predikciju. include_unranked=True
    vraca staro ponasanje (svi borci u kategoriji).

    Default active_since koristi SPLIT_DATE, pa se pri promjeni SPLIT_DATE mijenja
    i granica za "aktivne" borce u recommenderu.
    """
    source_df = pd.read_csv(DATA_PATH)
    fighter_profile = find_latest_fighter_profile(source_df, fighter_name)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model nije pronadjen: {model_path.resolve()}. "
            "Prvo pokreni: python zavrsni.py --train"
        )

    model = load_prediction_model(model_path)
    all_profiles = get_latest_profiles_by_weight_class(source_df)
    if all_profiles.empty:
        raise ValueError("Nije moguce izgraditi profile boraca iz dataseta.")

    fighter_key = fighter_profile["fighter_name"].casefold().strip()
    same_weight_class = all_profiles["weight_class"] == fighter_profile["weight_class"]
    not_same_fighter = all_profiles["fighter_name"].astype(str).str.casefold().str.strip() != fighter_key
    candidates = all_profiles.loc[same_weight_class & not_same_fighter].copy()

    if active_since is not None:
        active_since = pd.Timestamp(active_since)
        candidates = candidates.loc[candidates["latest_date"] >= active_since].copy()

    if not include_unranked:
        if "weightclass_rank" not in candidates.columns:
            raise ValueError(
                "Dataset nema rang stupac (match_weightclass_rank); "
                "pokreni s --include-unranked."
            )
        candidates = candidates.loc[candidates["weightclass_rank"].notna()].copy()

    if candidates.empty:
        filter_hint = (
            " Nema rangiranih protivnika; probaj --include-unranked."
            if not include_unranked
            else ""
        )
        raise ValueError(
            f"Nema kandidata za {fighter_profile['fighter_name']} "
            f"u kategoriji {fighter_profile['weight_class']}.{filter_hint}"
        )

    recommendations = []
    for _, opponent_profile in candidates.iterrows():
        opponent_profile = opponent_profile.to_dict()

        fighter_as_red_probability = predict_red_probability(
            model,
            fighter_profile,
            opponent_profile,
        )
        fighter_as_blue_probability = 1 - predict_red_probability(
            model,
            opponent_profile,
            fighter_profile,
        )
        fighter_probability = (
            fighter_as_red_probability + fighter_as_blue_probability
        ) / 2
        opponent_probability = 1 - fighter_probability

        recommendations.append(
            {
                "fighter": fighter_profile["fighter_name"],
                "opponent": opponent_profile["fighter_name"],
                "weight_class": fighter_profile["weight_class"],
                "opponent_rank": opponent_profile.get("weightclass_rank", float("nan")),
                "fighter_win_probability": fighter_probability,
                "opponent_win_probability": opponent_probability,
                # 1.00 = savrseno izjednacena borba (50/50), 0.00 = potpuno jednostrana.
                "competitiveness_score": 1 - 2 * abs(fighter_probability - 0.5),
                "distance_from_50_50": abs(fighter_probability - 0.5),
                "fighter_as_red_probability": fighter_as_red_probability,
                "fighter_as_blue_probability": fighter_as_blue_probability,
                "opponent_latest_profile_date": opponent_profile["latest_date"].date().isoformat(),
            }
        )

    recommendations_df = (
        pd.DataFrame(recommendations)
        .sort_values(["competitiveness_score", "opponent"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("PREPORUKA KOMPETITIVNIH MATCHUPOVA")
    print("=" * 70)
    fighter_rank_str = format_weightclass_rank(
        fighter_profile.get("weightclass_rank", float("nan"))
    )
    print(f"Fighter: {fighter_profile['fighter_name']} ({fighter_rank_str})")
    print(f"Weight class: {fighter_profile['weight_class']}")
    print(f"Model: {model_path}")
    if active_since is not None:
        print(f"Active filter: latest profile since {active_since.date()}")
    else:
        print("Active filter: disabled")
    print(
        "Rank filter: "
        + ("svi borci (ukljucujuci nerangirane)" if include_unranked else "samo rangirani (prvak + top 15)")
    )
    print(f"Top {len(recommendations_df)} najizjednacenijih protivnika:")

    for idx, row in recommendations_df.iterrows():
        print(
            f"{idx + 1:>2}. {row['fighter']} vs {row['opponent']} "
            f"({format_weightclass_rank(row['opponent_rank'])}) | "
            f"{row['fighter']}: {row['fighter_win_probability']:.4f}, "
            f"{row['opponent']}: {row['opponent_win_probability']:.4f}, "
            f"competitiveness: {row['competitiveness_score']:.4f}"
        )

    return recommendations_df


# ---------------------------------------------------------------------------
# Analiza i vizualizacija rezultata (--analyze)
# ---------------------------------------------------------------------------

def _select_result(
    results_df: pd.DataFrame, exp_prefix: str, model_name: str, scope: str
) -> pd.Series | None:
    """Dohvati jedan redak iz rezultatne tablice po eksperimentu, modelu i scope-u."""
    mask = (
        results_df["experiment"].astype(str).str.startswith(exp_prefix)
        & (results_df["model"] == model_name)
        & (results_df["evaluation_scope"] == scope)
    )
    matched = results_df.loc[mask]
    return None if matched.empty else matched.iloc[0]


def _annotate_bars(ax, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        height = bar.get_height()
        if pd.isna(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def compute_competitiveness_distribution(
    source_df: pd.DataFrame,
    model,
    active_since: str | pd.Timestamp | None = SPLIT_DATE,
    ranked_only: bool = True,
) -> pd.Series:
    """
    Izracunaj competitiveness score za sve moguce parove boraca unutar iste
    tezinske kategorije (neutralizirano na kut).

        competitiveness_score = 1 - 2 * |P(pobjede) - 0.5|
        1.00 = savrseno izjednacena borba, 0.00 = potpuno jednostrana

    Koristi se za histogram u --analyze: koliko je matchupova blizu 50/50.
    Default ranked_only=True gleda samo rangirane borce (prvak + top 15),
    konzistentno s default filtrom u recommend_matchups().
    """
    profiles = get_latest_profiles_by_weight_class(source_df)
    if active_since is not None:
        profiles = profiles.loc[
            profiles["latest_date"] >= pd.Timestamp(active_since)
        ]
    if ranked_only and "weightclass_rank" in profiles.columns:
        profiles = profiles.loc[profiles["weightclass_rank"].notna()]

    rows_a_red = []  # borac A kao Red: features = B - A
    rows_b_red = []  # borac B kao Red: features = A - B
    for _, group in profiles.groupby("weight_class"):
        records = group.to_dict("records")
        for a, b in itertools.combinations(records, 2):
            rows_a_red.append({f: b[f] - a[f] for f in FEATURE_COLS})
            rows_b_red.append({f: a[f] - b[f] for f in FEATURE_COLS})

    if not rows_a_red:
        return pd.Series(dtype=float)

    x_a_red = pd.DataFrame(rows_a_red, columns=FEATURE_COLS)
    x_b_red = pd.DataFrame(rows_b_red, columns=FEATURE_COLS)

    a_as_red_prob = model.predict_proba(x_a_red)[:, 1]
    a_as_blue_prob = 1 - model.predict_proba(x_b_red)[:, 1]
    a_prob = (a_as_red_prob + a_as_blue_prob) / 2

    scores = 1 - 2 * np.abs(a_prob - 0.5)
    return pd.Series(scores, name="competitiveness_score")


def analyze_results(model_path: Path | str = DEFAULT_PREDICTION_MODEL_PATH) -> None:
    """
    Generiraj grafove za zavrsni rad iz vec spremljenih rezultata i modela.

    Ne trenira ponovno; koristi results/model_results.csv, ufc_model_dataset.csv
    i spremljeni RF A2 model. Sve slike se spremaju u results/figures/.
    """

    import matplotlib
    

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Rezultati nisu pronadjeni: {RESULTS_PATH.resolve()}. "
            "Prvo pokreni: python zavrsni.py --train"
        )
    if not MODEL_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Model dataset nije pronadjen: {MODEL_DATASET_PATH.resolve()}. "
            "Prvo pokreni: python zavrsni.py --train"
        )
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model nije pronadjen: {model_path.resolve()}. "
            "Prvo pokreni: python zavrsni.py --train"
        )

    results_df = pd.read_csv(RESULTS_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = load_prediction_model(model_path)
    saved = []

    print("\n" + "=" * 70)
    print("ANALIZA I VIZUALIZACIJA REZULTATA")
    print("=" * 70)

    # Stvarni n-ovi iz rezultatne tablice (ne hardkodirano; prati SPLIT_DATE).
    n_full = int(results_df.loc[results_df["evaluation_scope"] == "full_test", "n_test"].iloc[0])
    n_odds = int(results_df.loc[results_df["evaluation_scope"] == "odds_available_test", "n_test"].iloc[0])

    # -----------------------------------------------------------------
    # Fig 1: no-odds modeli vs trivijalni baselinei (full test)
    # -----------------------------------------------------------------
    fig1_specs = [
        ("Always-Blue", ("Baseline", "Always-Blue", "full_test")),
        ("Always-Red", ("Baseline", "Always-Red", "full_test")),
        ("Model A\n(RF)", ("MODEL A:", "Random Forest", "full_test")),
        ("Model A2\n(RF, augment.)", ("MODEL A2", "Random Forest", "full_test_augmented_train")),
    ]
    labels, acc_vals, bal_vals = [], [], []
    for label, (exp, mdl, scope) in fig1_specs:
        row = _select_result(results_df, exp, mdl, scope)
        labels.append(label)
        acc_vals.append(row["accuracy"] if row is not None else np.nan)
        bal_vals.append(row["balanced_accuracy"] if row is not None else np.nan)

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, acc_vals, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, bal_vals, width, label="Balanced accuracy", color="#DD8452")
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="Slucajni pogodak (0.5)")
    _annotate_bars(ax, bars1)
    _annotate_bars(ax, bars2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rezultat")
    ax.set_ylim(0, 0.8)
    ax.set_title(f"Modeli bez odds-a vs trivijalni baselinei (test od {SPLIT_DATE.date()}, n={n_full})")
    ax.legend()
    fig.tight_layout()
    path1 = FIGURES_DIR / "fig1_accuracy_no_odds.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    saved.append(path1)

    # -----------------------------------------------------------------
    # Fig 2: ROC-AUC vs trziste (isti odds-podskup, n=1412)
    # -----------------------------------------------------------------
    fig2_specs = [
        ("Odds-Favorite\n(baseline)", ("Baseline", "Odds-Favorite", "odds_available_test")),
        ("Model A2 (RF)\nbez odds", ("MODEL A2", "Random Forest", "odds_available_test_augmented_train")),
        ("Model B (LR)\n+ odds", ("MODEL B", "Logistic Regression", "odds_available_test")),
        ("Model C (LR)\n+ odds/EV", ("MODEL C", "Logistic Regression", "odds_available_test")),
    ]
    labels2, roc_vals = [], []
    for label, (exp, mdl, scope) in fig2_specs:
        row = _select_result(results_df, exp, mdl, scope)
        labels2.append(label)
        roc_vals.append(row["roc_auc"] if row is not None else np.nan)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#C44E52", "#8172B3", "#4C72B0", "#55A868"]
    bars = ax.bar(labels2, roc_vals, color=colors, width=0.6)
    _annotate_bars(ax, bars)
    market_roc = roc_vals[0]
    if not pd.isna(market_roc):
        ax.axhline(
            market_roc,
            linestyle="--",
            color="#C44E52",
            linewidth=1,
            label=f"Linija trzista ({market_roc:.3f})",
        )
        ax.legend()
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 0.85)
    ax.set_title(f"ROC-AUC na istom odds-podskupu (n={n_odds}): model dostize, ne nadmasuje trziste")
    fig.tight_layout()
    path2 = FIGURES_DIR / "fig2_roc_auc_vs_market.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    saved.append(path2)

    # -----------------------------------------------------------------
    # Fig 3: feature importance RF A2
    # -----------------------------------------------------------------
    forest = model.named_steps["model"]
    importance = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": forest.feature_importances_})
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(importance["feature"], importance["importance"], color="#4C72B0")
    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w:.3f}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Vaznost (Gini importance)")
    ax.set_title("Vaznost znacajki — Random Forest, Model A2 (bez odds)")
    fig.tight_layout()
    path3 = FIGURES_DIR / "fig3_feature_importance.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    saved.append(path3)

    # -----------------------------------------------------------------
    # Fig 4: confusion matrix RF A2 na test skupu
    # -----------------------------------------------------------------
    model_df = pd.read_csv(MODEL_DATASET_PATH)
    model_df["date"] = pd.to_datetime(model_df["date"], errors="coerce")
    test_df = model_df.loc[model_df["date"] >= SPLIT_DATE]
    y_true = test_df["target"]
    y_pred = model.predict(test_df[FEATURE_COLS])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    class_labels = ["Blue (0)", "Red (1)"]
    ax.set_xticks([0, 1], labels=class_labels)
    ax.set_yticks([0, 1], labels=class_labels)
    ax.set_xlabel("Predvidjeno")
    ax.set_ylabel("Stvarno")
    ax.set_title("Confusion matrix — Random Forest, Model A2 (test)")
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=12,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path4 = FIGURES_DIR / "fig4_confusion_matrix.png"
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    saved.append(path4)

    # -----------------------------------------------------------------
    # Fig 5: distribucija competitiveness scorea (rangirani parovi u kategoriji,
    # konzistentno s default filtrom recommendera)
    # -----------------------------------------------------------------
    source_df = pd.read_csv(DATA_PATH)
    scores = compute_competitiveness_distribution(source_df, model, active_since=SPLIT_DATE)
    if scores.empty:
        print("Fig 5 preskocen: nema dovoljno aktivnih profila za parove.")
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(scores, bins=25, color="#55A868", edgecolor="white")
        mean_score = float(scores.mean())
        ax.axvline(
            mean_score,
            linestyle="--",
            color="#C44E52",
            linewidth=1.5,
            label=f"Prosjek = {mean_score:.3f}",
        )
        ax.set_xlabel("Competitiveness score  (1.00 = 50/50, 0.00 = jednostrano)")
        ax.set_ylabel("Broj mogucih matchupova")
        ax.set_title(
            f"Distribucija kompetitivnosti rangiranih parova u kategoriji "
            f"(n={len(scores)} parova)"
        )
        ax.legend()
        fig.tight_layout()
        path5 = FIGURES_DIR / "fig5_competitiveness_distribution.png"
        fig.savefig(path5, dpi=150)
        plt.close(fig)
        saved.append(path5)

    # -----------------------------------------------------------------
    # Fig 6: reliability diagram (kalibracija vjerojatnosti) za default model
    # -----------------------------------------------------------------
    # Koristi test predikcije iz fig4 (y_true, test_df su vec ucitani).
    test_probabilities = model.predict_proba(test_df[FEATURE_COLS])[:, 1]
    brier = brier_score_loss(y_true, test_probabilities)

    n_bins = 8
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_index = np.clip(
        np.digitize(test_probabilities, bin_edges) - 1, 0, n_bins - 1
    )
    # Binovi s manje od 20 primjera se izostavljaju: njihova stvarna stopa je
    # statisticki sum i vizualno bi iskrivila krivulju kalibracije.
    min_bin_count = 20
    bin_pred_means, bin_true_rates, bin_counts = [], [], []
    for b in range(n_bins):
        mask = bin_index == b
        if mask.sum() < min_bin_count:
            continue
        bin_pred_means.append(test_probabilities[mask].mean())
        bin_true_rates.append(y_true.values[mask].mean())
        bin_counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Savrsena kalibracija")
    ax.plot(
        bin_pred_means,
        bin_true_rates,
        marker="o",
        color="#4C72B0",
        label="Random Forest A2",
    )
    for x_val, y_val, count in zip(bin_pred_means, bin_true_rates, bin_counts):
        ax.annotate(
            f"n={count}",
            xy=(x_val, y_val),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=7,
            color="gray",
        )
    ax.set_xlabel("Predvidjena vjerojatnost (Red pobjeda)")
    ax.set_ylabel("Stvarna stopa Red pobjeda")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Reliability diagram — Random Forest, Model A2 (test)\n"
        f"Brier score = {brier:.4f} (0.25 = neinformativan)"
    )
    ax.legend()
    fig.tight_layout()
    path6 = FIGURES_DIR / "fig6_reliability_diagram.png"
    fig.savefig(path6, dpi=150)
    plt.close(fig)
    saved.append(path6)

    print(f"\nSpremljeno {len(saved)} slika u: {FIGURES_DIR.resolve()}")
    for path in saved:
        print(f"  - {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_and_evaluate() -> None:
    model_df = load_and_prepare_data(DATA_PATH)

    all_results: list[dict] = []

    # Baselines — uvijek-Red, uvijek-Blue, odds-favorite
    evaluate_baselines(model_df, all_results)

    # Eksperiment A: samo dif featurei, bez odds
    run_experiment(
        model_df,
        "MODEL A: OSNOVNE DIF ZNAČAJKE, BEZ ODDS",
        FEATURE_COLS,
        "without_odds",
        all_results,
    )

    # Eksperiment A2: isti featurei kao Model A, ali train skup je corner-augmented
    run_experiment(
        model_df,
        "MODEL A2: OSNOVNE DIF ZNACAJKE, BEZ ODDS, CORNER AUGMENTATION",
        FEATURE_COLS,
        "augmented_without_odds",
        all_results,
        augment=True,
    )

    # Eksperiment B: dif + R_odds + B_odds (bez EV)
    run_experiment(
        model_df,
        "MODEL B: DIF ZNAČAJKE + R_ODDS + B_ODDS (bez EV)",
        FEATURE_COLS + MONEYLINE_ODDS_COLS,
        "with_moneyline_odds",
        all_results,
    )

    # Eksperiment C: dif + R_odds + B_odds + R_ev + B_ev
    print(
        "\nNapomena za Model C: R_ev i B_ev su deterministicka transformacija "
        "moneyline odds-a, pa se ne ocekuje velika razlika u odnosu na Model B."
    )
    run_experiment(
        model_df,
        "MODEL C: DIF ZNAČAJKE + ODDS + EV",
        FEATURE_COLS + ODDS_EV_COLS,
        "with_odds_ev",
        all_results,
    )

    # Usporedna tablica
    save_results_table(all_results)

    print("\n" + "=" * 70)
    print("Svi eksperimenti uspješno završeni.")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "UFC AI model: treniranje modela ili brza predikcija hipotetske borbe."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--train",
        action="store_true",
        help="Treniraj modele, evaluiraj ih i spremi rezultate.",
    )
    mode.add_argument(
        "--predict",
        nargs=2,
        metavar=("FIGHTER_A", "FIGHTER_B"),
        help=(
            "Predvidi hipotetsku borbu bez ponovnog treniranja modela. "
            "Default je neutralan na redoslijed/kutove."
        ),
    )
    mode.add_argument(
        "--recommend",
        metavar="FIGHTER_NAME",
        help="Predlozi najkompetitivnije protivnike za zadanog borca.",
    )
    mode.add_argument(
        "--analyze",
        action="store_true",
        help="Generiraj grafove za rad iz spremljenih rezultata (results/figures/).",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_PREDICTION_MODEL_PATH),
        help=(
            "Putanja do spremljenog .pkl modela za predikciju. "
            f"Default: {DEFAULT_PREDICTION_MODEL_PATH}"
        ),
    )
    parser.add_argument(
        "--allow-different-weight-class",
        action="store_true",
        help="Dopusti predikciju i ako borci nemaju istu zadnju poznatu kategoriju.",
    )
    parser.add_argument(
        "--respect-corners",
        action="store_true",
        help=(
            "Nemoj neutralizirati kutove; prvo ime tretiraj kao Red, drugo kao Blue."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Broj preporucenih matchupa za --recommend. Default: 10",
    )
    parser.add_argument(
        "--active-since",
        default=str(SPLIT_DATE.date()),
        help=(
            "Za --recommend uzmi samo protivnike ciji je zadnji profil od ovog datuma. "
            f"Default: {SPLIT_DATE.date()}"
        ),
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Za --recommend nemoj filtrirati neaktivne/stare borce.",
    )
    parser.add_argument(
        "--include-unranked",
        action="store_true",
        help=(
            "Za --recommend ukljuci i nerangirane protivnike "
            "(default su samo rangirani: prvak + top 15)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train:
        train_and_evaluate()
        return

    if args.predict:
        fighter_a, fighter_b = args.predict
        predict_fight(
            fighter_a,
            fighter_b,
            model_path=args.model,
            enforce_same_weight_class=not args.allow_different_weight_class,
            neutralize_corner=not args.respect_corners,
        )
        return

    if args.recommend:
        recommend_matchups(
            args.recommend,
            top_n=args.top,
            model_path=args.model,
            active_since=None if args.include_inactive else args.active_since,
            include_unranked=args.include_unranked,
        )
        return

    if args.analyze:
        analyze_results(model_path=args.model)
        return


if __name__ == "__main__":
    main()
