# Observations — S&P 500 Semantic Map

## What separates along each axis?

The horizontal axis (Old Economy ↔ New Economy / Tech) does exactly what you'd expect once you see the plot. Technology and Communication Services companies cluster hard to the right — Nvidia, Apple, Microsoft, Google (Alphabet) all land there, which makes sense because their names carry decades of cultural baggage around "software" and "digital." Energy and Utilities sit on the left, which also tracks — ExxonMobil, Chevron, and the utility companies are linguistically associated with words like "extraction," "infrastructure," and "pipes," even if they've been trying to rebrand themselves. Industrials and Materials split somewhere in the middle, which feels about right since those sectors straddle the old/new boundary depending on the specific company.

The vertical axis (Defensive / Stable ↔ Speculative / High-Growth) is where things get more interesting. Consumer Staples companies (Coca-Cola, Procter & Gamble, Kimberly-Clark) all score toward the defensive end — the model seems to pick up on the fact that these brand names are synonymous with boring, everyday reliability. Utilities land even lower, which aligns with how investors actually think about them: you buy Duke Energy for the dividend, not the moonshot potential. On the other end, some Consumer Discretionary names like Tesla and Carvana score toward the speculative/growth end, again matching how those companies are actually talked about in financial media.

## Most surprising finding

The most unexpected result is where Goldman Sachs ends up — it sits closer to the defensive end than you might guess given how associated the name is with Wall Street risk-taking and trading. The embedding model seems to weight the "established institution" connotation of the name more than the "high-finance speculation" connotation, which says something about how language encodes reputation versus behavior. Similarly, Boeing shows up more toward the industrial/old-economy side than the high-tech side, even though it's an aerospace and defense company that builds some of the most technically complex products on earth. The model doesn't know that — it only knows how the word "Boeing" appears in text, and apparently "Boeing" reads as heavy manufacturing, not cutting-edge tech.

## What would a third axis capture?

A third axis worth exploring would be **domestic vs. global / multinational**. Something like "international operations, global expansion" vs. "local regional business, domestic market." This would probably separate companies like JPMorgan, Procter & Gamble, and Exxon (which are genuinely global operations) from regional banks, local utilities, and domestic-only retailers. It would also pull apart some sectors that currently overlap — Real Estate is almost entirely domestic, while Materials and Energy have significant international exposure. That distinction doesn't show up in either of the two axes we built, but it's a real structural divide in how these companies actually operate.

---
*Assignment: Sprint M04 — Semantic Axes | Dataset: S&P 500 (203 companies, 11 sectors) | Model: all-MiniLM-L6-v2*
