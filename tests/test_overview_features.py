from __future__ import annotations

from oneehr.data.overview_light import build_feature_overview


def test_build_feature_overview_formats_display_names():
    out = build_feature_overview(
        dynamic_feature_columns=["num__Age", "cat__Sex__M", "cat__Sex__F", "cat__Race__Asian"],
        static_feature_columns=["num__BMI"],
        top_k_categoricals=10,
    )

    dyn = out["dynamic"]
    assert dyn["n_numeric"] == 1
    assert dyn["n_categorical"] == 3
    assert dyn["numeric_features"] == ["Age"]

    # Categorical bases should be grouped (Sex has 2 levels).
    bases = {x["name"]: x["n_levels"] for x in dyn["categorical_feature_bases"]}
    assert bases["Sex"] == 2
    assert bases["Race"] == 1

