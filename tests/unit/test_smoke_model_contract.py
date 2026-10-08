import torch

from scripts.smoke_model_contract import _clone_tree


def test_clone_tree_recursively_detaches_and_clones_nested_tensors() -> None:
    source = {
        "mapping": {0: torch.tensor([1.0])},
        "sequence": [torch.tensor([2.0]), (torch.tensor([3.0]),)],
    }
    cloned = _clone_tree(source)

    assert cloned["mapping"][0].device.type == "cpu"
    assert cloned["mapping"][0].data_ptr() != source["mapping"][0].data_ptr()
    assert cloned["sequence"][0].data_ptr() != source["sequence"][0].data_ptr()
    assert cloned["sequence"][1][0].data_ptr() != source["sequence"][1][0].data_ptr()
    torch.testing.assert_close(cloned["sequence"][1][0], torch.tensor([3.0]))
