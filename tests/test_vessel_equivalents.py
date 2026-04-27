import numpy as np
import pandas as pd
import pytest

from vascx_models.vessel_widths import (
    compute_revised_crx_from_widths,
    select_vessel_width_measurements_for_equivalents,
)


def _vessel_width_record(
    connection_index: int,
    sample_index: int,
    width_px: float,
    vessel_type: str = "artery",
    image_id: str = "sample",
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "inner_circle": "2r",
        "outer_circle": "3r",
        "inner_circle_radius_px": 40.0,
        "outer_circle_radius_px": 60.0,
        "connection_index": connection_index,
        "sample_index": sample_index,
        "x": float(connection_index),
        "y": float(sample_index),
        "width_px": width_px,
        "x_start": 0.0,
        "y_start": 0.0,
        "x_end": 1.0,
        "y_end": 1.0,
        "vessel_type": vessel_type,
    }


def _vessel_tortuosity_record(
    connection_index: int,
    tortuosity: float,
    vessel_type: str = "artery",
    image_id: str = "sample",
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "inner_circle": "2r",
        "outer_circle": "3r",
        "inner_circle_radius_px": 40.0,
        "outer_circle_radius_px": 60.0,
        "connection_index": connection_index,
        "x_start": 0.0,
        "y_start": 0.0,
        "x_end": 1.0,
        "y_end": 1.0,
        "path_length_px": tortuosity,
        "chord_length_px": 1.0,
        "tortuosity": tortuosity,
        "vessel_type": vessel_type,
    }


def test_compute_revised_crx_from_widths_records_selected_vessel_ids() -> None:
    records = []
    for vessel_type, widths in {
        "artery": [10.0, 14.0, 8.0],
        "vein": [20.0, 18.0],
    }.items():
        for connection_index, width in enumerate(widths, start=1):
            for sample_index in range(1, 3):
                records.append(
                    _vessel_width_record(
                        connection_index,
                        sample_index,
                        width,
                        vessel_type,
                    )
                )
    df_widths = pd.DataFrame.from_records(records)

    df_connections, df_equivalents = compute_revised_crx_from_widths(df_widths)
    selected = select_vessel_width_measurements_for_equivalents(
        df_widths,
        df_connections,
    )

    assert df_equivalents["metric"].tolist() == ["CRAE", "CRVE"]
    assert df_equivalents["requested_n_largest"].tolist() == [6, 6]
    assert df_equivalents["n_vessels_available"].tolist() == [3, 2]
    assert df_equivalents["n_vessels_used"].tolist() == [3, 2]
    assert df_equivalents["vessel_ids_used"].tolist() == [
        "artery_2;artery_1;artery_3",
        "vein_1;vein_2",
    ]
    assert np.isfinite(df_equivalents["equivalent_px"]).all()
    assert set(selected["vessel_type"]) == {"artery", "vein"}
    assert len(selected) == len(df_widths)


def test_compute_revised_crx_from_widths_limits_selection_to_six_largest() -> None:
    records = []
    for connection_index in range(1, 9):
        for sample_index in range(1, 3):
            records.append(
                _vessel_width_record(
                    connection_index=connection_index,
                    sample_index=sample_index,
                    width_px=float(connection_index),
                    vessel_type="artery",
                )
            )
    df_widths = pd.DataFrame.from_records(records)

    df_connections, df_equivalents, rounds = compute_revised_crx_from_widths(
        df_widths,
        return_rounds=True,
    )
    selected = select_vessel_width_measurements_for_equivalents(
        df_widths,
        df_connections,
    )

    assert df_equivalents.iloc[0]["n_vessels_available"] == 8
    assert df_equivalents.iloc[0]["n_vessels_used"] == 6
    assert df_equivalents.iloc[0]["vessel_ids_used"] == (
        "artery_8;artery_7;artery_6;artery_5;artery_4;artery_3"
    )
    selected_connection_ids = set(
        df_connections.loc[df_connections["selected_for_equivalent"], "connection_index"]
    )
    assert selected_connection_ids == {
        3,
        4,
        5,
        6,
        7,
        8,
    }
    assert set(selected["connection_index"]) == {3, 4, 5, 6, 7, 8}
    assert len(selected) == 12
    assert ("sample", "artery") in rounds


def test_compute_revised_crx_from_widths_aggregates_selected_tortuosity() -> None:
    width_records = []
    tortuosity_records = []
    for connection_index in range(1, 9):
        for sample_index in range(1, 3):
            width_records.append(
                _vessel_width_record(
                    connection_index=connection_index,
                    sample_index=sample_index,
                    width_px=float(connection_index),
                    vessel_type="artery",
                )
            )
        tortuosity_records.append(
            _vessel_tortuosity_record(
                connection_index=connection_index,
                tortuosity=1.0 + connection_index / 10.0,
                vessel_type="artery",
            )
        )
    df_widths = pd.DataFrame.from_records(width_records)
    df_tortuosities = pd.DataFrame.from_records(tortuosity_records)

    df_connections, df_equivalents = compute_revised_crx_from_widths(
        df_widths,
        df_tortuosities=df_tortuosities,
    )

    selected_connection_ids = df_connections.loc[
        df_connections["selected_for_equivalent"], "connection_index"
    ].tolist()
    assert selected_connection_ids == [3, 4, 5, 6, 7, 8]
    assert df_equivalents.iloc[0]["mean_tortuosity_used"] == pytest.approx(1.55)


def test_compute_revised_crx_from_widths_reports_single_vessel_without_equivalent() -> None:
    df_widths = pd.DataFrame.from_records(
        [
            _vessel_width_record(1, 1, 12.0, "vein"),
            _vessel_width_record(1, 2, 12.0, "vein"),
        ]
    )

    df_connections, df_equivalents = compute_revised_crx_from_widths(df_widths)
    selected = select_vessel_width_measurements_for_equivalents(
        df_widths,
        df_connections,
    )

    row = df_equivalents.iloc[0]
    assert row["metric"] == "CRVE"
    assert row["n_vessels_available"] == 1
    assert row["n_vessels_used"] == 1
    assert row["vessel_ids_used"] == "vein_1"
    assert np.isnan(row["equivalent_px"])
    assert len(selected) == 2


def test_compute_revised_crx_from_widths_preserves_empty_output_schema() -> None:
    empty_widths = pd.DataFrame(
        columns=[
            "image_id",
            "inner_circle",
            "outer_circle",
            "inner_circle_radius_px",
            "outer_circle_radius_px",
            "connection_index",
            "sample_index",
            "x",
            "y",
            "width_px",
            "x_start",
            "y_start",
            "x_end",
            "y_end",
            "vessel_type",
        ]
    )

    df_connections, df_equivalents = compute_revised_crx_from_widths(empty_widths)
    selected = select_vessel_width_measurements_for_equivalents(
        empty_widths,
        df_connections,
    )

    assert df_connections.empty
    assert df_connections.columns.tolist() == [
        "image_id",
        "vessel_type",
        "connection_index",
        "vessel_id",
        "inner_circle",
        "outer_circle",
        "inner_circle_radius_px",
        "outer_circle_radius_px",
        "mean_width_px",
        "n_samples",
        "selected_for_equivalent",
    ]
    assert df_equivalents.empty
    assert df_equivalents.columns.tolist() == [
        "image_id",
        "metric",
        "vessel_type",
        "requested_n_largest",
        "n_vessels_available",
        "n_vessels_used",
        "vessel_ids_used",
        "mean_widths_used_px",
        "equivalent_px",
    ]
    assert selected.empty
    assert selected.columns.tolist() == empty_widths.columns.tolist()
