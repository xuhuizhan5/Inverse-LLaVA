# Adding a benchmark

1. Confirm that it answers a preregistered question and identify the official
   split, license, annotations, scorer, and external dependency.
2. Add one strict YAML config. Include the model's image placeholder in the
   executable `prompt_template` and declare its conversation template. Reuse
   declarative multiple-choice/VQA behavior; add Python only if official
   semantics differ.
3. Add a tiny non-copyright fixture, expected predictions, official scorer output,
   coverage/leakage tests, and a benchmark card.
4. Confirm the generated protocol ID changes when the prompt, extraction,
   scorer, generation, split, or source content changes. Prepared example and
   prediction manifests bind the exact sample set separately. Mark the protocol
   golden-verified only after exact agreement with the official scorer.
5. Keep generation records generic so another checkpoint can use the same adapter.
