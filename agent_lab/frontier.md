# Agent Lab Frontier

This file is the lab's current one-page theory of the model.

Use it for the question: what does the evidence now say the architecture wants?

Use [`state.md`](./state.md) for the live dashboard, [`findings.md`](./findings.md) for durable claims, and [`experiments.tsv`](./experiments.tsv) for exact runs.

## Current Thesis

The model does not want a uniform small transformer. It wants a depth-specialized hybrid architecture.

Best current thesis:

- the lower stack should do cheaper structured sequence processing
- the upper stack should keep richer global reasoning
- FFNs are still doing real work and cannot be broadly collapsed
- block recipes do not need to stay perfectly uniform if the non-uniformity is periodic and well-placed
- skip topology matters, but skip parameterization can be simplified
- dynamic routing is only useful in a very narrow top-of-stack form
- broad simplification stories usually fail unless they are carefully targeted

The current best valid run, [`AL-20260331-017`](./experiments.tsv), fits that picture exactly:

- lower four attention layers replaced by stronger mixers
- upper attention stack preserved
- shared scalar skip gates added on top
- dense untied output head retained

So the winning move is not "make everything smaller." It is "stop paying for the same kind of computation at every depth."

## What The Model Seems To Want

### 1. Cheaper lower processing

Lower-stack attention was overbuilt. Replacing the lower four attention layers with sequence mixers was the clearest architectural win the lab has found. But pushing farther than that usually hurts, which means the real lesson is selective simplification, not blanket attention removal.

### 2. Repeated upper global refresh

The upper stack still wants real global reasoning. Plain local-window replacement lost. Collapsing the upper stack to only one or two global reasoners also lost. The best simplification signal in this region is [`AL-20260401-059`](./experiments.tsv): interleaving upper attention with lighter refinement nearly tied the frontier. That suggests the upper stack may need periodic global refresh, not dense full attention in every upper block.

### 3. Real dense FFNs in most places

Tranche `Y` answered the childlike FFN question clearly: broad FFN minimalism is wrong. No-expand MLPs failed badly, and even half-width FFNs lost clearly. The model still needs true `project up -> nonlinear transform -> project down` FFNs in most of the stack. The only surviving nuance is that lower mixer-heavy layers may tolerate somewhat lighter FFNs than upper attention-heavy layers.

### 4. Simple routing, not broad routing

Routing matters, but the winning form is simpler than expected. Shared scalar skip gates stack cleanly with the hybrid backbone and produced the current best run. Broad late-layer AttnRes-lite collapsed. Narrow top-only routing survived and even came close as a secondary family, which suggests routing complexity is only worth spending at the final representation-selection stage.

### 5. Strong output path, conservative changes elsewhere

The output head was an early frontier family and is still part of the winning recipe. Dense untied outputs, tighter softcap, and faster head LR were real gains. In contrast, many later "clever" replacements for core backbone pieces lost. The model seems happy to spend bytes on strong output mapping and real FFNs, but not on naive compression-native simplifications.

### 6. Local lower-stage sharing may exist, but only with preserved layer identity

The first whole-block sharing story failed. But the second-generation sharing branch with small deltas changed the picture: pairwise lower-mixer sharing stayed close to the frontier, while broader upper-band or FFN sharing did not. That suggests the lower mixer stage may contain some real reusable structure, but only if each layer still gets a small amount of individuality back.

## Strong Conclusions

- Lower-four hybrid mixers are real and remain the main frontier family.
- Shared scalar skip gates are the cleanest routing win on top of that backbone.
- Broad late-layer dynamic routing is wrong; top-only narrow routing is the only surviving AttnRes-lite story.
- Plain local-window replacement of the remaining upper attention layers is wrong on the current hybrid backbone.
- `relu^2` remains the correct broad MLP anchor.
- Cubic-heavy polynomial MLPs are wrong.
- Broad FFN minimalism is wrong.
- Naive low-rank factorization is not the first compression-native win.
- Naive whole-block sharing is also not the first compression-native win.
- Naive token-selective heavy/light compute is also not the next efficiency breakthrough.

## Near-Survivors

These are not frontier wins, but they changed the theory and still deserve to shape future work.

- [`AL-20260331-035`](./experiments.tsv): top-only narrow AttnRes-lite plus skip gates. This keeps top-only dynamic routing alive as a secondary family.
- [`AL-20260331-043`](./experiments.tsv): mixed linear-plus-quadratic MLP. This is the only polynomial MLP variant that stayed credibly near the frontier.
- [`AL-20260331-040`](./experiments.tsv): gated SiLU MLP was competitive on quality but invalid on size. Gating is not dead; it is a size-control question.
- [`AL-20260401-049`](./experiments.tsv): lower-light / upper-full FFN structure. This is the only real survivor from broad FFN-minimalism.
- [`AL-20260401-055`](./experiments.tsv): periodic heavy/light blocks. This is the strongest clue that block recipes may want periodic concentration rather than full uniformity.
- [`AL-20260401-059`](./experiments.tsv): interleaved upper attention and lighter refinement. This is the strongest current clue that the upper stack may be simplifiable without collapse.
- [`AL-20260401-077`](./experiments.tsv): pairwise lower-mixer sharing with deltas. This is the strongest new compression-native near-survivor.
- [`AL-20260401-080`](./experiments.tsv): lower-mixer sharing plus reallocated mixer width. This keeps the “share then reallocate” story alive.

## Dead Stories

These stories are not impossible forever, but they are now poor default bets.

- "Just add depth" without recovering optimizer steps
- broad residual simplification
- broad late-layer AttnRes-lite
- plain local-window replacement of the upper attention stack
- naive low-rank factorization as the compression-native breakthrough
- naive whole-block sharing as the compression-native breakthrough
- naive token-selective heavy attention or joint heavy-token routing
- broad smooth-family MLP replacement
- cubic-heavy polynomial MLPs
- broad no-expand or strongly shrunken FFNs
- collapsing the upper stack to one or two global reasoners

## Resolution

The lab now has a real architecture picture:

- simplify the lower stack
- preserve multiple chances for upper global reasoning
- keep FFNs real
- keep skips, but simplify how they are weighted
- use dynamic routing only in a very narrow top-of-stack form
- stop looking for one blunt compression trick that saves everything at once

The key repeated lesson is:

**carefully placed simplicity wins; broad complexity and broad collapse both lose**

That is the current scientific resolution of the program.

## Live Questions

- Is [`AL-20260331-017`](./experiments.tsv) close to locally saturated, or is there still headroom in mixer strength or routing strength?
- Is [`AL-20260401-059`](./experiments.tsv) a real upper-stack simplification path or just a near-miss?
- Is [`AL-20260401-055`](./experiments.tsv) a real clue that periodic heavy/light block concentration is better than uniform blocks?
- Can [`AL-20260401-077`](./experiments.tsv) and [`AL-20260401-080`](./experiments.tsv) be turned into a real compression-native branch?
- What actually broke the latent upper-reasoner branch, and does it become scientifically interesting once repaired?
