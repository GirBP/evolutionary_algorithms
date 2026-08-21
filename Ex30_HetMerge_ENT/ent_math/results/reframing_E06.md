# Reframing Analysis: Task Vectors vs Permutation Alignment

## Why Task Vectors Failed on Toy (E03-E06)

The task vector approach requires a STRONG pretrained base (W_base) such that
τ_A = W_A - W_base and τ_B = W_B - W_base capture ONLY the task-specific changes.

In the toy setting:
- Base trained for only 5 epochs → weak, underfitting representation
- τ_A and τ_B capture BOTH feature learning AND task specialization
- Adding τ_A + τ_B overshoots because each vector includes redundant representation changes

Result: cos(τ_A, τ_B) ≈ 0 (orthogonal) but magnitude too large → destructive interference.

## Why Permutation Alignment Worked (E01-E02)

The alignment approach bypasses the base model entirely:
- Directly aligns channels between W_A and W_B
- Merges aligned weights with per-layer λ
- BN reset recalibrates statistics

Result: 4/4 retention (100%).

## Reframing

For toy models WITHOUT a pretrained base:
→ Use permutation alignment + EA (proven 4/4).

For ResNet-18 WITH pretrained base (ImageNet):
→ Task vectors MAY work because the base model is strong.
→ But the toy results suggest alignment is more robust.

## Decision

Proceed to ResNet-18 with the ALIGNMENT approach (proven).
If time allows, compare with task vectors on ResNet-18 (strong base).
