import torch

from invllava.model.fusion import FusionBlock
from invllava.model.llama.configuration import LlamaArchitecture
from invllava.model.llama.generation import generate
from invllava.model.llama.modeling import InverseLlamaForCausalLM
from invllava.model.sequence import build_text_only_fusion_state
from invllava.model.types import ExpandedSequence


def build_model() -> InverseLlamaForCausalLM:
    config = LlamaArchitecture(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    fusion = FusionBlock(
        hidden_size=16,
        visual_size=8,
        target_sizes={"q": 16, "k": 16, "v": 16},
    )
    return InverseLlamaForCausalLM(config, fusion_layers={0: fusion}).eval()


def test_cached_last_token_matches_full_forward() -> None:
    torch.manual_seed(3)
    model = build_model()
    ids = torch.tensor([[1, 7, 8, 9]])
    full_state = build_text_only_fusion_state(model.model.embed_tokens(ids), 8)
    full = model(input_ids=ids, fusion_state=full_state)

    prefix = ids[:, :3]
    prefix_state = build_text_only_fusion_state(model.model.embed_tokens(prefix), 8)
    first = model(input_ids=prefix, fusion_state=prefix_state, use_cache=True)
    last = ids[:, 3:]
    last_state = build_text_only_fusion_state(model.model.embed_tokens(last), 8)
    cached = model(
        input_ids=last,
        attention_mask=torch.ones((1, 4), dtype=torch.bool),
        fusion_state=last_state,
        past_key_values=first.past_key_values,
        use_cache=True,
    )
    torch.testing.assert_close(full.logits[:, -1], cached.logits[:, -1], atol=1e-5, rtol=1e-5)


def test_left_padded_batch_generation_matches_individual_generation() -> None:
    torch.manual_seed(7)
    model = build_model()

    def initial(ids: torch.Tensor, mask: torch.Tensor) -> ExpandedSequence:
        embeddings = model.model.embed_tokens(ids)
        return ExpandedSequence(
            inputs_embeds=embeddings,
            attention_mask=mask,
            position_ids=mask.long().cumsum(-1).sub(1).clamp_min(0),
            labels=None,
            fusion_state=build_text_only_fusion_state(embeddings, 8, mask),
        )

    batched_ids = torch.tensor([[0, 1, 7], [1, 8, 9]])
    batched_mask = torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.bool)
    batched = generate(
        model,
        initial(batched_ids, batched_mask),
        max_new_tokens=2,
        eos_token_id=999,
    )
    first_ids = torch.tensor([[1, 7]])
    first = generate(
        model,
        initial(first_ids, torch.ones_like(first_ids, dtype=torch.bool)),
        max_new_tokens=2,
        eos_token_id=999,
    )
    second_ids = torch.tensor([[1, 8, 9]])
    second = generate(
        model,
        initial(second_ids, torch.ones_like(second_ids, dtype=torch.bool)),
        max_new_tokens=2,
        eos_token_id=999,
    )
    assert torch.equal(batched[0], first[0])
    assert torch.equal(batched[1], second[0])


def test_full_recompute_generation_matches_cached_greedy_tokens() -> None:
    torch.manual_seed(11)
    model = build_model()
    ids = torch.tensor([[1, 7, 8, 9]])
    embeddings = model.model.embed_tokens(ids)
    initial = ExpandedSequence(
        inputs_embeds=embeddings,
        attention_mask=torch.ones_like(ids, dtype=torch.bool),
        position_ids=torch.arange(ids.shape[1]).unsqueeze(0),
        labels=None,
        fusion_state=build_text_only_fusion_state(embeddings, 8),
    )
    cached = generate(model, initial, max_new_tokens=3, eos_token_id=999)
    recomputed = generate(
        model,
        initial,
        max_new_tokens=3,
        eos_token_id=999,
        cache_mode="full_recompute",
    )
    assert torch.equal(cached, recomputed)
