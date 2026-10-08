from invllava.eval.records import PredictionRecord, PredictionStore


def test_prediction_store_resumes_and_rejects_duplicates(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    store = PredictionStore(path, protocol_id="p", checkpoint_id="c")
    record = PredictionRecord(1, "p", "e", "c", "s", "prompt", "answer")
    store.append(record)
    resumed = PredictionStore(path, protocol_id="p", checkpoint_id="c")
    assert resumed.completed == {"s"}
    try:
        resumed.append(record)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate record was accepted")
