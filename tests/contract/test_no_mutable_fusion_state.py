from invllava.model.llama.configuration import LlamaArchitecture
from invllava.model.llama.modeling import InverseLlamaForCausalLM


def test_model_has_no_request_scoped_mutable_vision_state() -> None:
    model = InverseLlamaForCausalLM(
        LlamaArchitecture(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    names = {name for name, _ in model.named_modules()}
    assert all("vision_hidden_states" not in name for name in names)
    assert not hasattr(model, "vision_hidden_states")
