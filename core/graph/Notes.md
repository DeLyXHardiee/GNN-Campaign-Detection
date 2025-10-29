# Things to consider
### Adding in-reply-to edges between email nodes
Possible benefits:
Phishing campaigns sometimes reuse message IDs or construct fake reply chains (“RE: invoice”) to appear legitimate. Linking by thread reveals such behavior.
When labels appear only on one message in a thread, label propagation to its replies/forwards becomes trivial.

Possible drawbacks:
If our dataset contains only isolated reports, we will rarely have valid In-Reply-To headers so adding these edges will produce mostly missing or zero-degree connections, offering little structural value and some overhead.
Worse, if users forward phishing emails (not reply), the In-Reply-To will point to their own message, not the original phish, so edges may be meaningless.

