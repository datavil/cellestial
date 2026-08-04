# How cellestial's deferred layers actually work

This explains the trick behind

```python
umap + cl.arrow_axis()          # new way: plot is never passed explicitly
```

versus the old

```python
umap + cl.arrow_axis(umap)      # old way: plot passed twice, effectively
```

Read it top-to-bottom. Each section builds on the previous one.

---

## 1. Why layers need the plot at all

Lets-Plot's built-in layers (e.g. `geom_point()`) don't need the plot. They just
declare "draw points using aesthetics x and y." The actual columns and data are
supplied by the enclosing `ggplot(data, mapping)`, and Lets-Plot wires them up
at render time.

Cellestial's layers are different. They have to **compute data from the plot's
data** before drawing:

- `cluster_outlines` runs a kernel density estimate on the UMAP points.
- `bracket` runs pairwise statistical tests.
- `arrow_axis` reads the data extent to position the arrows.
- `stream` extracts velocity columns and runs streamplot.

They can't just declare aesthetics and hope for the best — they need the
underlying DataFrame and the `x`/`y` column names up front. That's why the old
API forced you to pass `plot` explicitly.

**Analogy:** a normal Lets-Plot layer is like a cookie-cutter — you hand it
dough (the plot's data) at bake time. A cellestial layer is like a recipe
that needs to *inspect your pantry* before it even knows what it's making.
So somehow, the recipe has to get a look inside the pantry.

---

## 2. What `a + b` actually does in Python

When you write `a + b`, Python doesn't just call one method. It runs a small
protocol:

1. Try `a.__add__(b)`. Three outcomes:
   - Returns a value → done, that's the result.
   - Returns `NotImplemented` → move to step 2.
   - **Raises `TypeError`** → the whole expression raises. **No step 2.**
2. Try `b.__radd__(a)` (the "reflected add"). Same three outcomes.
3. If both return `NotImplemented` → `TypeError: unsupported operand type(s)`.

Notice step 1's middle case: `NotImplemented` is a **polite shrug** that
says "I don't know how to add this, let the other operand try." Raising
`TypeError` is a **slammed door** that ends the whole thing.

**Analogy:** you call your friend to borrow a book (`a.__add__(b)`). If they
say "I don't have it, try Bob" — that's `NotImplemented`; you call Bob
(`b.__radd__(a)`). If they hang up on you — that's `TypeError`; the protocol
stops, Bob never gets called.

---

## 3. The first idea that was wrong

My first instinct was: make `cl.arrow_axis()` return a tiny proxy object with
a `__radd__` method.

```python
class DeferredLayer:
    def __init__(self, builder):
        self._builder = builder

    def __radd__(self, plot):
        return plot + self._builder(plot)
```

The theory was:

1. `plot + deferred` → Python calls `PlotSpec.__add__(deferred)`.
2. `PlotSpec` doesn't know what a `DeferredLayer` is → returns `NotImplemented`.
3. Python falls back to `deferred.__radd__(plot)`.
4. Inside `__radd__`, we finally have `plot`, so we build the real layer and
   return `plot + real_layer`.

Clean. Elegant. **And broken.**

When I actually ran it, I got:

```
TypeError: unsupported operand type(s) for +: PlotSpec and DeferredLayer
```

---

## 4. Why it was broken

Look at what Lets-Plot's `PlotSpec.__add__` does with an unknown type:

```python
def __add__(self, other):
    if isinstance(other, DummySpec): ...
    elif isinstance(other, FeatureSpec): ...
    # ... handled cases ...
    return super().__add__(other)    # ← falls through to FeatureSpec.__add__
```

And `FeatureSpec.__add__` (its parent class) ends with:

```python
raise TypeError('unsupported operand type(s) for +: ...')
```

**Slammed door.** Not `NotImplemented`. The `__radd__` fallback never fires,
because Python's protocol explicitly says: *TypeError is not a shrug, it's a
hard stop.* This is almost certainly a Lets-Plot bug (or at least an
infelicity) — by convention, `__add__` should return `NotImplemented` for
unknown types. But we don't own Lets-Plot, so we need a workaround.

**Analogy:** we called the friend hoping for "try Bob." They hung up instead.
Now we need a different way to reach Bob.

---

## 5. The escape hatch: subclass priority

Python's operator protocol has one extra rule I glossed over in section 2.
The full rule is:

> If `type(b)` is a **subclass** of `type(a)`, then Python tries
> `b.__radd__(a)` **first**, before `a.__add__(b)`.

This is called **reflected method priority**, and it exists precisely so that
subclasses can override the behavior of their parent's operators — even when
the subclass is on the right-hand side.

Read that again. If `DeferredLayer` is a *subclass of `PlotSpec`*, then
`plot + deferred` calls `deferred.__radd__(plot)` **before** touching
`PlotSpec.__add__`. We skip the slammed door entirely.

So the fix is:

```python
class DeferredLayer(PlotSpec):       # ← note: subclass of PlotSpec
    def __init__(self, builder):
        FeatureSpec.__init__(self, kind="deferred", name=None)  # see §6
        self._builder = builder

    def __radd__(self, plot):
        return plot + self._builder(plot)
```

**Analogy:** your friend hangs up on strangers, but they always answer calls
from family. So we pretend to be family. Now when you call
`plot + deferred`, Python sees "family member" and routes the call to
`deferred.__radd__` first. The friend never gets a chance to hang up.

---

## 6. Why we bypass `PlotSpec.__init__`

`PlotSpec.__init__` requires real plot arguments: `data`, `mapping`, `scales`,
`layers`, and so on. But our `DeferredLayer` **has none of these** — it's a
placeholder, not a real plot. Calling the full `PlotSpec.__init__` would
force us to invent dummy values that would then get interpreted as real plot
state later.

So instead, we skip `PlotSpec.__init__` entirely and call the grandparent's
`FeatureSpec.__init__` directly:

```python
FeatureSpec.__init__(self, kind="deferred", name=None)
```

This gives us just enough of the `FeatureSpec` machinery to satisfy `isinstance`
checks and not much else. The object quacks like a `PlotSpec` for the one
purpose we care about (the subclass priority rule in §5) and stays empty
otherwise.

**Analogy:** we want a fake ID that says "family" on the outside so the
operator protocol lets us in. But we don't want to actually *be* a plot
— we have no data, no layers, no scales. So we skip the ceremony
(`PlotSpec.__init__`) and just stamp the minimum ID card
(`FeatureSpec.__init__`).

The only method on `DeferredLayer` that ever gets called is `__radd__`.
Nothing else should touch it. If someone tries to render a `DeferredLayer`
directly or call `.get_plot_shared_data()` on it, it'll break — but that's
fine; nobody should.

---

## 7. The full sequence: `umap + cl.cluster_outlines(groups="B Cells")`

Putting it all together, here's exactly what happens, step by step:

1. **`cl.cluster_outlines(groups="B Cells")` runs first.**
   The function captures `groups` in a closure and returns
   `DeferredLayer(builder)`. No data is touched yet. No KDE is computed.
   If you never add this to a plot, nothing happens. It's an IOU.

2. **Python evaluates `umap + deferred`.**
   - `type(umap)` is `PlotSpec`.
   - `type(deferred)` is `DeferredLayer`, which subclasses `PlotSpec`.
   - Subclass priority kicks in → Python calls `deferred.__radd__(umap)`
     **before** `umap.__add__(deferred)`.

3. **Inside `__radd__(umap)`:**
   - `self._builder(umap)` runs. This is where the closure finally
     sees the plot:
     - `umap.get_plot_shared_data()` → pulls the Polars DataFrame.
     - `get_mapping(umap)` → extracts `x`, `y`, `color` column names.
     - `_get_density_boundaries(...)` → runs the KDE.
     - Returns a real `LayerSpec` (a `geom_path` over the boundary points).
   - Then `return umap + real_layer` runs.

4. **`umap + real_layer` is a normal Lets-Plot addition.**
   - `type(real_layer)` is `FeatureSpec` (not a subclass of `PlotSpec`), so
     no priority flip. `PlotSpec.__add__` handles it the normal way and
     returns a new `PlotSpec` with the layer appended.

5. **That new `PlotSpec` is the final result.**
   It's the value of `umap + cl.cluster_outlines(groups="B Cells")`.

**Analogy:** you order a custom cake (`cl.cluster_outlines(groups=...)`). The
bakery gives you a receipt, not a cake — it doesn't know what flour to use
yet. Later you walk into a kitchen (`+ umap`). The kitchen recognizes the
receipt, reads your pantry, bakes the cake with your flour, and hands you
the finished cake.

---

## 8. The `plot=` escape hatch

One quirk: sometimes you want to draw the layer on a **different** plot than
the one supplying the data. The canonical example is:

```python
gggrid([plot, ggplot() + cl.stream(plot=plot)])
```

Here, the stream lines are computed from `plot`'s UMAP + velocity columns,
but they're rendered on an empty `ggplot()` canvas beside it.

To support this, every layer accepts an optional keyword-only `plot=...`
argument:

```python
def cluster_outlines(groups, *, plot=None, ...):
    explicit_plot = plot
    def _build(receiving_plot):
        source = explicit_plot if explicit_plot is not None else receiving_plot
        # ... use `source` for data and mapping ...
    return DeferredLayer(_build)
```

- When `plot=None` (the common case), the builder uses `receiving_plot` — the
  plot on the left side of `+`.
- When `plot=umap` is passed explicitly, the builder uses it regardless of
  what's on the left side of `+`.

Both paths still go through `DeferredLayer`. The `+` operator is always what
triggers construction — `plot=` just pins the data source ahead of time.

---

## 9. What the tradeoffs are

The deferred pattern is a net win, but it's not free.

### Loses eager error surfacing

Under the old API, `cl.cluster_outlines(plot, groups="BogusGroup")` would
raise immediately — the KDE ran at call time. Under the new API, the same
mistake only raises when you do `plot + cl.cluster_outlines(groups="BogusGroup")`.
The stack trace is one level deeper, and the error point is the `+`, not
the call.

Usually fine, occasionally annoying in Jupyter when a cell's `cl.bracket(...)`
call looks OK but the next cell's `violin + bracket_ref` blows up.

### Can't compose layers without a plot

With real `LayerSpec` objects, `layer1 + layer2` produces a `FeatureSpecArray`
that you can reuse. With `DeferredLayer`, neither operand has a plot to
introspect, so `cl.cluster_outlines(...) + cl.arrow_axis()` fails.

Workaround: always start with a plot. `plot + cl.a() + cl.b()` works —
the first `+` produces a real `PlotSpec`, and the second `+` then sees that
`PlotSpec` on its left side.

### `DeferredLayer` isn't really a `PlotSpec`

It pretends to be one so that Python's operator protocol routes calls to our
`__radd__`. But it has no data, no layers, no scales. Don't call
`.get_plot_shared_data()`, `.as_dict()`, or rendering methods on it —
they'll either blow up or return garbage.

This is a contained lie. The only safe thing to do with a `DeferredLayer` is
add it to a plot.

---

## 10. Recap in one paragraph

Lets-Plot's `PlotSpec.__add__` raises `TypeError` for unknown right-hand
operands instead of returning `NotImplemented`, so the usual Python
`__radd__` fallback is dead on arrival. We route around it by making
`DeferredLayer` a subclass of `PlotSpec`, which triggers Python's
**reflected operator priority rule**: when the right operand is a subclass
of the left operand, `__radd__` is tried *before* `__add__`. Our `__radd__`
gets the plot handed to it, runs the closure-captured builder against the
plot's data and aesthetics, builds the real `LayerSpec`, and returns
`plot + real_layer`. We bypass `PlotSpec.__init__` because the wrapper holds
no real plot state; `FeatureSpec.__init__` is enough to satisfy isinstance
checks. That's the whole trick.
