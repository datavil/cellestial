
# Project Instructions

## Preferences
- Keep explanations brief and to the point.
- Ask questions when necessary for complex tasks.
- think before coding: ask when unsure, don't silently pick one interpretation and run with it
- simplicity first: minimum code, any overengineering shows at a glance
- surgical edits: only touch what's required, don't fix up neighboring code on the way by
- goal-driven: translate fuzzy instructions into verifiable targets before starting

## Code Style
- use polars over pandas.
- use ruff style formatting & import sorting, Numpy docstrings style.
- In polars, use lazyframes when they will provide performance benefits.
- use lets-plot over matplotlib or seaborn.
- modularity is crucial.
- use expressive naming for variables and parameters such as "group_index" over "grp_idx". 
- Avoid abbreviations for variable and parameter names unless it is well-known convention such as "n" for sample size "X" for matrix and similar.
- prefer naming dataframes "frame" over "df".
- Cellestial uses poetry for package management and virtual environment.
- In general follow the best practises around the python ecosystem.
- prefer robustness over flexibility.
- just like Rust, Python functions should preferably return single type.

## Principles
### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

LLMs often pick an interpretation silently and run with it. This principle forces explicit reasoning:

- **State assumptions explicitly** — If uncertain, ask rather than guess
- **Present multiple interpretations** — Don't pick silently when ambiguity exists
- **Push back when warranted** — If a simpler approach exists, say so
- **Stop when confused** — Name what's unclear and ask for clarification

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

Combat the tendency toward overengineering:

- No features beyond what was asked
- No abstractions for single-use code
- No "flexibility" or "configurability" that wasn't requested
- No error handling for impossible scenarios
- If 200 lines could be 50, rewrite it

**The test:** Would a senior engineer say this is overcomplicated? If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting
- Don't refactor things that aren't broken
- Match existing style, even if you'd do it differently
- If you notice unrelated dead code, mention it — don't delete it

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused
- Don't remove pre-existing dead code unless asked

**The test:** Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform imperative tasks into verifiable goals:

| Instead of... | Transform to... |
|--------------|-----------------|
| "Add validation" | "Write tests for invalid inputs, then make them pass" |
| "Fix the bug" | "Write a test that reproduces it, then make it pass" |
| "Refactor X" | "Ensure tests pass before and after" |

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```


