from __future__ import annotations

import pandas as pd


def _clean_feature_name(name: str) -> str:
    clean = name.replace("num__", "").replace("cat__", "")
    clean = clean.replace("_", " ")
    return clean.title()


def top_feature_contributions(model, row: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    preprocess = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]

    transformed = preprocess.transform(row)
    feature_names = preprocess.get_feature_names_out()
    coefficients = estimator.coef_[0]
    contributions = transformed[0] * coefficients

    output = pd.DataFrame(
        {
            "Feature": [_clean_feature_name(name) for name in feature_names],
            "Contribution": contributions,
        }
    )
    output["Direction"] = output["Contribution"].apply(
        lambda value: "Raises fraud risk" if value > 0 else "Lowers fraud risk"
    )
    output["Impact"] = output["Contribution"].abs()
    output = output.sort_values("Impact", ascending=False).head(limit)
    output["Contribution"] = output["Contribution"].round(3)
    output["Impact"] = output["Impact"].round(3)
    return output[["Feature", "Direction", "Contribution", "Impact"]]
