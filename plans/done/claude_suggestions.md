## ~~Documentation & Discoverability~~
~~The philosophy is clear internally, but new users won't have read it. A one-page "philosophy" section in the docs (like Polars has) would help users understand why the API looks the way it does, which builds trust and reduces frustration when they hit the intentional constraints.~~
## Error Messages as a First-Class Feature
Given the emphasis on explicitness and predictability, error messages should match that standard. When a user passes a sequence to umap() instead of umaps(), the error should tell them exactly that — "did you mean umaps()?" Rust-inspired errors would be a natural fit here given the stated influences.
## ~~Type Hints & IDE Support~~
~~The singular/plural return type guarantee (PlotSpec vs SupPlotsSpec) is only as useful as the tooling that surfaces it. Strict, complete type annotations would let IDEs and type checkers (mypy, pyright) enforce the guarantees the philosophy promises, making the design philosophy machine-checkable, not just a convention.~~
## Benchmarks as Documentation
~~Since performance is explicitly a selling point, publishing reproducible benchmarks against comparable tools (Scanpy's plotting, Squidpy, etc.) at different dataset sizes would validate the Polars choice and give users confidence before adopting it.~~
## ~~Escape Hatches~~
~~The strict, opinionated design is a strength but will occasionally frustrate power users. A documented, explicit way to access the intermediate Polars DataFrame before it hits Lets-Plot would let advanced users customize without fighting the library — and it aligns with the modularity principle already in the philosophy.~~
## Consistency Auditing
As the library grows, the singular/plural convention and the no-abbreviation rule will face pressure. A short internal style guide or a linting rule that enforces naming conventions would protect the philosophy from gradually eroding as more contributors join.

The __overarching suggestion__ is: the philosophy is strong enough to be a public-facing asset, not just an internal guide. Leaning into it explicitly in docs, errors, and tooling support would differentiate Cellestial meaningfully in a crowded visualization space.