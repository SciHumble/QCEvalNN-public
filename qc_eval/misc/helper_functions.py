from datetime import datetime
import pandas as pd
from qc_eval.misc.parameters import EmbeddingType


def convert_datetime_to_string(date: datetime) -> str:
    """
    Converts a datetime into a string that can be used to store a file with
    a time stamp.
    Args:
        date: datetime

    Returns:
        YYYY-MM-DD-HH-MM
    """
    result = date.strftime("%Y-%m-%d-%H-%M")
    return result


def quantum_autosafe_file(row: pd.Series) -> str:
    date: str = row["date"]
    # Format: '2025-02-13 08:35:26.674902'

    input_size: int = row["input_size"]
    file: str = f"qcnn-{input_size}-{date[:10]}-{date[11:13]}-{date[14:16]}.pt"

    return file


def quantum_embedding(row: pd.Series) -> str:
    compacted = row["compacted_dataset"]
    if compacted:
        return EmbeddingType.angle.value
    else:
        return EmbeddingType.angle_compact.value

