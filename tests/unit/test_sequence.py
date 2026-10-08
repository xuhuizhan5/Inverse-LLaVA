import torch

from invllava.model.sequence import build_text_only_fusion_state, expand_multimodal_sequence


def test_placeholder_expands_to_patch_stream() -> None:
    input_ids = torch.tensor([[10, -200, 11]])
    embeddings = torch.randn(1, 3, 8)
    visual = torch.randn(4, 6)
    expanded = expand_multimodal_sequence(
        input_ids=input_ids,
        token_embeddings=embeddings,
        image_features=[[visual]],
        image_token_id=-200,
        labels=input_ids.clone(),
    )
    assert expanded.inputs_embeds.shape == (1, 6, 8)
    assert expanded.fusion_state.visual_features.shape == (1, 6, 6)
    assert expanded.fusion_state.text_mask.tolist() == [[True, False, False, False, False, True]]
    assert expanded.labels.tolist()[0][1:5] == [-100] * 4
    assert torch.equal(expanded.fusion_state.visual_features[0, 1:5], visual)


def test_placeholder_image_mismatch_fails() -> None:
    input_ids = torch.tensor([[10, -200]])
    try:
        expand_multimodal_sequence(
            input_ids=input_ids,
            token_embeddings=torch.randn(1, 2, 4),
            image_features=[[]],
            image_token_id=-200,
        )
    except ValueError as error:
        assert "placeholder" in str(error)
    else:
        raise AssertionError("mismatch did not fail")


def test_text_only_fusion_mask_excludes_padding() -> None:
    hidden = torch.randn(2, 3, 4)
    attention = torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.bool)
    state = build_text_only_fusion_state(hidden, 6, attention)
    assert torch.equal(state.text_mask, attention)
    assert not state.vision_mask.any()


def test_multimodal_truncation_rejects_sample_without_target_tokens() -> None:
    input_ids = torch.tensor([[-200, 10]])
    try:
        expand_multimodal_sequence(
            input_ids=input_ids,
            token_embeddings=torch.randn(1, 2, 4),
            image_features=[[torch.randn(3, 6)]],
            image_token_id=-200,
            labels=torch.tensor([[-100, 10]]),
            max_length=3,
        )
    except ValueError as error:
        assert "no supervised token" in str(error)
    else:
        raise AssertionError("truncation that removes every target token should fail")


def test_expansion_allows_masked_sample_beside_supervised_sample() -> None:
    input_ids = torch.tensor([[1, 10, 11], [1, 12, 13]])
    expanded = expand_multimodal_sequence(
        input_ids=input_ids,
        token_embeddings=torch.randn(2, 3, 4),
        image_features=[[], []],
        image_token_id=-200,
        labels=torch.tensor([[-100, -100, -100], [-100, 12, 13]]),
        visual_feature_dim=6,
    )

    assert expanded.labels is not None
    assert expanded.labels[0].eq(-100).all()
    assert expanded.labels[1].ne(-100).any()
