import pytest

from tau2.agent.jlens_failure_protocol import validate_official_airline_splits


def test_failure_protocol_preserves_official_train_test_boundary():
    official = {"train": ["0", "1", "2"], "test": ["3", "4"]}
    matrix = {
        "splits": {
            "train_task_ids": ["0", "1"],
            "validation_task_ids": ["2"],
            "evaluation_task_ids": ["3", "4"],
        }
    }

    validate_official_airline_splits(matrix, official)

    matrix["splits"]["evaluation_task_ids"] = ["2", "4"]
    with pytest.raises(ValueError, match="official test"):
        validate_official_airline_splits(matrix, official)
