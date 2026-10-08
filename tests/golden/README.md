# Golden protocol tests

Add only small, redistributable expected outputs here. Required goldens are:
Vicuna-v1 token IDs/response masks, CLIP preprocessing and selected hidden shape,
one manuscript-checkpoint logit/generation fixture, clean save/reload parity, and
one official scorer fixture for every benchmark config. Record upstream revision
and fixture license beside each file. Generate expected values with the pinned
upstream implementation and retain the comparison record.
