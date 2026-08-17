# Public Equity: AI Concentration

## Consolidated conclusion

AI concentration in public equity is already high, but it must be separated into three different concepts: **gross benchmark footprint**, **economic dependence on AI**, and **concentration of the underlying economic drivers**. The evidence is strongest for the first and third; there is not yet enough company-level data to produce a defensible aggregate economic-exposure percentage.

For the S&P 500, the best current conclusion is that a **broad AI-linked cohort represents approximately 44–50% of index market capitalisation**. This range reconciles JPMorgan's 30-stock JPAMAIDE basket, which represented 44% of the index in October 2025, with the Bank of England's broader AI-fund-holdings-based basket, which represented approximately 50% on 25 June 2026. A narrower, locally reproducible seven-issuer cohort represented **29.82%** of an S&P 500 tracker in August 2026. The seven-name figure is useful as a transparent reference, but it is not a lower bound because the selection rule is incomplete. [Methodology transparency note](ai_reponse/ai-source-methodology-transparency2.md) · [BoE, July 2026 FSR](https://www.bankofengland.co.uk/financial-stability-report/2026/july-2026) · [Seven-name calculation](ai_reponse/ChatGPT_DeepSearch.md)

The corresponding broad MSCI World exposure cannot yet be calculated from the available materials. The same seven US issuers represented **20.31%** of MSCI World in July 2026, but extrapolating the S&P 500's 44–50% range into MSCI World would not be defensible without a global constituent-level basket. Japan demonstrates how strongly the result can depend on the benchmark: a JPMorgan-curated 50-company AI, semiconductor and data-centre cohort represented **33.7% of TOPIX 500** and **54.8% of Nikkei 225**, and was expected to generate approximately 59% of FY26 profit growth. [JPMorgan Japan Equity Strategy, p. 4](research/JP_Morgan-Japan_Equity_Strategy_Nikkei_225_KOSPI_Correlation_and_Outlook-2026-07-27.pdf)

The broader investment implication is more important than the exact market-cap percentage. Technology and Communication Services supplied **54% of S&P 500 earnings growth in 1Q26**. JPMorgan estimates AI-related companies represent **59% of projected 2026 S&P capex**, while hyperscalers account for approximately **87% of AI capex in both 2026 and 2027**. Therefore, the listed beneficiary universe can broaden into utilities, industrials, power and adopters while control of the investment cycle remains concentrated in a small number of platform companies. [JPMorgan, pp. 6–8](research/JP_Morgan-The_JP_Morgan_View_All_the_Rest_is_No_AI-2026-05-08.pdf) · [JPMorgan, pp. 1–2](research/JP_Morgan-US_Equity_Strategy_Market_Update_Earnings_AI_Capex_and_Monetization-2026-08-10.pdf)

## 1. What is AI concentration today?

### Definitions

For public equity, AI concentration should be reported through three separate measures:

1. **Gross benchmark footprint:** the market-cap weight of companies included in a named AI cohort. This is observable and reproducible when the cohort and benchmark constituents are available, but it treats a diversified company as 100% AI-linked.
2. **Economic AI exposure:** benchmark weight multiplied by the proportion of each company's revenue, profit, backlog, assets, load or capacity that is economically dependent on AI. This is conceptually superior but disclosure remains incomplete and denominators differ by business model.
3. **Driver concentration:** the degree to which the total exposure depends on a small number of companies, customers or capex decision-makers. This is measured using top-five/top-ten shares, HHI, effective number of drivers and hyperscaler share of AI capex.

### Current estimate

| Universe and measure | Current result | Evidence treatment |
|---|---:|---|
| S&P 500 broad gross AI-linked footprint | **44–50%** | Preferred current range. Source-reported; basket constituents are not locally available. |
| S&P 500 narrow seven-issuer footprint | **29.82%** | Derived calculation from displayed tracker weights; transparent but incomplete selection rule. |
| Effective issuers within the seven-name cohort | **5.55** | Derived from a conditional HHI of 0.180. This is concentration inside the cohort, not the HHI of the full index. |
| MSCI World narrow seven-issuer footprint | **20.31%** | Derived calculation; not a broad MSCI World AI estimate. |
| Japan 50-name cohort | **33.7% TOPIX 500; 54.8% Nikkei 225** | Source-reported analyst mapping; illustrates benchmark and definition sensitivity. |
| S&P 500 economic AI exposure | **Not yet measurable** | Missing company-level economic shares and complete basket memberships. |

The apparently conflicting S&P 500 estimates are therefore not estimates of the same object. The **29.82%** result covers seven selected issuers; **44%** covers JPMorgan's 30-stock basket; and **approximately 50%** uses the BoE's broader fund-holdings-derived definition. The correct treatment is to preserve these definitions and use **44–50% as the broad current gross-footprint range**, with 29.82% as a transparent narrow reference.

## 2. How should it be measured?

The most defensible framework is a three-layer model.

### Layer 1: source-defined gross footprint

For benchmark \(P\) and source basket \(s\):

\[
F_{P,s}=\sum_i w_{i,P}I_{i,s}
\]

where \(w_{i,P}\) is issuer \(i\)'s benchmark weight and \(I_{i,s}\) equals one when the issuer belongs to source basket \(s\). Share classes should be consolidated at the parent level. Results should be reported separately for each source, followed by the intersection and deduplicated union of compatible baskets. This makes methodological uncertainty observable rather than hiding it inside an internally invented “base case.” [Implementation plan](AI_Concentration_Tracker_Implementation_Plan.md)

GICS should describe the resulting cohort, not determine inclusion. The proposed Vanguard-style sleeve—Software & Services, Technology Hardware & Equipment, Semiconductors & Semiconductor Equipment and Electric Utilities—is useful only as a mechanical upper envelope because it includes many companies with little or no material AI exposure. The user's numerical GICS calculation is not present in the workspace, so it cannot yet be incorporated as a calculated result.

### Layer 2: evidence-adjusted economic exposure

For issuers with sufficient evidence:

\[
E_P=\sum_i w_{i,P}e_i
\]

where \(e_i\) is the sourced share of the company's economics attributable to AI. The evidence hierarchy should be business-model-specific:

- direct AI companies: AI-linked revenue, gross profit or operating profit;
- hyperscalers: AI/cloud revenue and usage, supported by capex and utilisation;
- semiconductor and equipment suppliers: AI/data-centre revenue, orders or backlog;
- utilities and power companies: contracted data-centre load, rate-base capex or profit contribution;
- data-centre real estate: contracted MW or rent, adjusted for tenant quality and project stage;
- adopters and disrupted companies: measurable cost, margin or revenue effects, with a positive or negative sign.

Unknown exposure should remain unknown. Revenue, backlog, capex, MW and PBT cannot be combined mechanically without a rule selecting the controlling denominator for each business model. Economic exposure should therefore be reported with evidence coverage, rather than as a false-precision point estimate.

### Layer 3: concentration and risk

Within the AI exposure, calculate each issuer's contribution \(c_i=w_i e_i\), its normalised share \(p_i=c_i/\sum c_i\), and:

\[
HHI=\sum_i p_i^2, \qquad N_{effective}=1/HHI
\]

Report market-cap footprint, economic exposure and return sensitivity separately. A factor regression can identify companies whose returns behave like the AI theme after controlling for market, sector, growth/value, rates and energy, but return beta is a risk measure—not an economic-exposure percentage. Morgan Stanley's thematic work is the most developed model in the source set: it combines analyst mapping, residual thematic returns, stock-level betas and a refined factor-risk overlay. Its main limitation is look-ahead in historical membership and potential contamination by conventional growth, momentum and duration factors. [Morgan Stanley, Thematic Tides](research/Morgan_Stanley-Thematic_Tides_Quantifying_the_Next_Big_Waves-2025-11-10.pdf) · [Morgan Stanley, Global Thematic Risk](research/Morgan_Stanley-Quant_Matters_Mapping_the_Thematic_Landscape_and_Risk_Attribution-2026-02-20.pdf)

## 3. What could concentration look like in 3–5 years?

### Source anchors

The research supports a positive near-term relative growth advantage for AI-linked companies, but not a precise five-year market-cap forecast:

- JPMorgan forecasts AI capex of approximately **$900bn in 2026** and **more than $1.2tn in 2027**, with hyperscalers approximately 87% of the total. [JPMorgan, p. 1](research/JP_Morgan-US_Equity_Strategy_Market_Update_Earnings_AI_Capex_and_Monetization-2026-08-10.pdf)
- AI-upstream companies supplied approximately **10% of S&P earnings growth in 2023**; JPMorgan forecasts approximately **54% in 2026** and **56% in 2027**, with AI-company earnings growth of roughly 20% over the next two years. [JPMorgan, p. 22](research/JP_Morgan-Macro_Corporate_Spotlight_Update_for_Corporate_Treasurers-2026-07-30.pdf)
- JPMorgan's latest S&P forecast implies approximately **15% EPS growth in 2027**. The gap to roughly 20% AI-company earnings growth provides a source anchor for a mid-single-digit relative-growth assumption. [JPMorgan, p. 1](research/JP_Morgan-US_Equity_Strategy_Market_Update_Earnings_AI_Capex_and_Monetization-2026-08-10.pdf)
- Goldman Sachs estimates approximately **$14–17tn** of AI-attributed market-value gains against a baseline present value of approximately **$9tn** for additional US AI capital revenues. This does not prove overvaluation, but it implies that continued concentration requires earnings delivery, global value capture or sustained valuation support. [Goldman Sachs, pp. 5–9](research/Goldman_Sachs-Global_Markets_Analyst_Where_the_AI_Boom_Stands_Now_Markets_Ahead_of_the_Macro-2026-06-22.pdf)

### Forward calculation

Let \(W_0\) be the current cohort weight and \(q\) its annual market-cap growth relative to the rest of the index. Holding cohort membership fixed:

\[
W_h=\frac{W_0(1+q)^h}{W_0(1+q)^h+(1-W_0)}
\]

This formula makes the required assumption explicit. It does not require a forecast of the absolute index level.

The base case assumes **4% annual relative market-cap growth**. This is an analyst assumption informed by the approximately five-percentage-point near-term earnings-growth advantage in the JPMorgan forecasts, with a modest allowance for valuation normalisation. The high-concentration case assumes **8% annual relative growth**, which requires stronger persistence and little offset from multiple compression. No probabilities are assigned.

| Scenario | Assumption | Narrow cohort: 29.82% today | Broad cohort: 44–50% today |
|---|---|---:|---:|
| No relative advantage | \(q=0\) | 29.8% in 3y and 5y | 44–50% in 3y and 5y |
| **Base / continued leadership** | \(q=+4\%\) annually | **32.3% in 3y; 34.1% in 5y** | **46.9–52.9% in 3y; 48.9–54.9% in 5y** |
| **High concentration** | \(q=+8\%\) annually | **34.9% in 3y; 38.4% in 5y** | **49.7–55.7% in 3y; 53.6–59.5% in 5y** |
| Capex/valuation reset | One-time 20% relative decline, then equal growth | 25.4% | 38.6–44.4% |

The resulting preferred forward view is:

> If AI-linked companies maintain a mid-single-digit relative market-cap growth advantage, the broad S&P 500 AI-linked footprint could rise from 44–50% today to approximately **47–53% in three years** and **49–55% in five years**. A stronger 8% annual relative advantage would produce approximately **50–56% in three years** and **54–60% in five years**.

Useful breakpoints are also explicit. The seven-name cohort must outperform the rest of the index by approximately **9.4% annually for five years** to rise from 29.82% to 40%. A 44% broad cohort needs approximately **4.9% annual relative growth** to reach 50% in five years; a 50% cohort needs approximately **8.4%** to reach 60%.

This is a sensitivity range, not a forecast produced by a completed constituent-level model. A market crash is not required for concentration to fall: positive AI earnings can coexist with declining index weight if non-AI earnings grow faster, AI valuation multiples compress, share issuance dilutes the cohort, or new benchmark entrants broaden the denominator.

The final model should calculate each constituent's future market capitalisation as:

\[
MarketCap_{i,t}=EPS_{i,t}\times DilutedShares_{i,t}\times(P/E)_{i,t}
\]

and then renormalise forecast market caps to benchmark weights. It should produce both a **frozen-cohort view**, which measures appreciation of today's AI group, and a **live-cohort view**, which admits new enablers, adopters and disrupted companies over time.

## 4. Evidence, assumptions and remaining gaps

The analysis is research-grade for the direction and approximate scale of concentration, but not yet decision-grade for one exact broad benchmark percentage.

The principal evidence and methodological limitations are:

- **Broad basket constituents are missing.** The BoE methodology is disclosed, but its company list is not. The JPAMAIDE 30 list is also absent, preventing local reconstruction of the 44–50% current range.
- **The BoE historical comparison has current-universe bias.** Its 2022, 2025 and 2026 observations use June 2026 constituents, so the move from roughly 25% to 50% measures appreciation of the current cohort rather than a contemporaneous historical index universe.
- **The seven-name calculation is transparent but not fully auditable.** Its arithmetic has been independently verified, but the exported AI response lost permanent source links for the underlying weights and does not document why several plausible AI names were omitted.
- **No broad MSCI World calculation exists.** The 20.31% figure covers only the seven selected US issuers. The Perplexity estimate of “40%+” is an unsupported extrapolation and should not be used.
- **The manual GICS result is absent.** Only the proposed four-industry-group methodology is present. No constituent table, calculation, code or numerical result exists in the workspace.
- **Economic-exposure data are incomplete.** There is no issuer-level table of AI revenue, profit, backlog, capex, load, MW or rent shares, nor a documented rule for reconciling those denominators.
- **Capex estimates use different scopes.** The available 2026 figures range from approximately $695–725bn of four-company guidance to roughly $900bn of broader AI capex. These are different issuer sets and accounting definitions, not contradictory estimates to be averaged.
- **The forward range is assumption-led.** The 4% and 8% annual relative-growth cases are explicit analyst sensitivities, not sell-side forecasts. The five-year result also assumes fixed membership, no benchmark methodology change and no dilution beyond the relative-growth input.

The most important data upgrades are a frozen point-in-time S&P 500 constituent file, at least one complete broad AI basket, the missing GICS calculation, company-level 2027–31 EPS/share-count/valuation assumptions, and a documented economic-exposure table. Until those are available, the appropriate conclusion is a transparent scenario range rather than a single precise forecast.
