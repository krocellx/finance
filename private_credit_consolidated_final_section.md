# Private Credit: AI concentration

*Evidence cut-off: 17 August 2026. Most portfolio-stock observations are for 3Q25 or year-end 2025.*

## Executive conclusion

Private credit has a **two-sided AI concentration**, not one additive “AI weight.” The best current estimate of **direct, positive AI financing** is **almost 8% of outstanding direct-loan volume, or more than $200bn**, in the matched PitchBook universe used by the BIS. Separately, **SaaS represents 19% of all direct loans** and software represents roughly **20% of BDC lending**; this is a broad **AI-disruption screen**, not evidence that every software loan is negatively exposed. Morgan Stanley’s 26% public-BDC estimate uses a narrower vehicle sample and a broader software-plus-IT-services definition. The positive and negative measures can overlap and must not be added.

The defensible 2030 range for the narrow, positive stock is **about 7–13% of private-credit AUM, with 10% as a midpoint**, calculated by dividing the BIS forecast of **$300–600bn** of outstanding AI-related loans by Preqin’s **$4.5tn** 2030 private-credit AUM forecast. This is a normalization, not an observed market forecast: the numerator and denominator have imperfectly aligned coverage. The important conclusion is that AI-linked private-credit dollars are likely to rise substantially, while the portfolio share need not rise nearly as quickly because the asset class itself is forecast to grow.

| Measure | Current or future estimate | Status | Appropriate interpretation |
|---|---:|---|---|
| Positive/buildout AI loan stock | **Almost 8%; >$200bn** | Observed estimate | Best current market headline; matched direct-loan numerator and denominator |
| SaaS disruption screen | **19%; >$500bn** of direct loans | Observed estimate | Broad vulnerability screen, not “AI exposure” and not additive to 8% |
| BDC software screen | **~20% / $115bn**; **26%** for public BDC software plus IT services | Observed estimates | Vehicle- and taxonomy-specific cross-checks; large manager dispersion |
| Positive/buildout AI loan stock in 2030 | **~7–13%; 10% midpoint** | Calculated scenario | $300–600bn BIS forecast divided by $4.5tn Preqin forecast |

## 1. What is AI concentration today?

### Definition and denominator

For private credit, “AI concentration” should mean the share of **gross credit exposure at risk from the same AI economic driver**, measured before fund leverage. It has two separate sleeves:

1. **Buildout exposure:** loans whose borrower, use of proceeds, collateral, cash flow or ultimate tenant is materially tied to AI compute, data centres, networking, power or an AI-native business.
2. **Disruption exposure:** loans to incumbents whose revenue or debt-service capacity could be impaired by AI substitution, price compression or higher product-investment needs. Software is the best available screen, but it is not a complete classification.

The current market headline uses the **outstanding direct-loan volume in the same PitchBook sample as both numerator and denominator**. The BIS defines an AI firm through PitchBook’s Artificial Intelligence, Big Data and Cloud Tech verticals. On that basis, AI-related loans rose to **more than $200bn and almost 8% of outstanding direct-loan volume**. Only about **20% of funds** had made an AI-related loan, and the average fund exposure was about **5%**, indicating that the aggregate exposure is uneven and weighted toward larger or more active lenders. Loan structures did not yet show an obvious AI premium: AI loans were 46% secured versus 48% for other loans, with average spreads of 6.2 percentage points versus 6.1 points. ([BIS Bulletin 120, January 2026](https://www.bis.org/publ/bisbull120.pdf))

This **8% is the preferred current estimate**, but it is narrow. A borrower-vertical taxonomy can miss data-centre, power, equipment and asset-backed loans made to diversified companies or special-purpose vehicles where AI is the economic use of proceeds.

### The separate disruption concentration

By year-end 2025, SaaS loans exceeded **$500bn, or 19% of all direct loans**, up from about $8bn in 2015. Roughly one-third of funds lent to SaaS. ([BIS Quarterly Review, March 2026](https://www.bis.org/publ/qtrpdf/r_qt2603v.htm)) In the more transparent BDC universe, software loans were about **$115bn, roughly one-fifth of all BDC lending and more than 80% of BDC technology portfolios**. The top five BDCs held 37% of software lending and the top ten held more than half; about 60% of software lending went to borrowers financed by seven or more BDCs. This is common-borrower concentration, even where funds appear diversified. ([BIS Bulletin 128, July 2026](https://www.bis.org/publ/bisbull128.htm))

Morgan Stanley found **26% software-plus-IT-services exposure across 42 public BDCs** at 3Q25, versus 23% for software alone and 19% in private-credit CLOs. Individual public BDC exposure ranged from low single digits to more than 50%. The 26% figure is higher than the BIS’s roughly 20% because it adds IT services and covers listed BDCs representing only about 30% of BDC AUM. The broader BIS estimate is therefore the better BDC headline; Morgan Stanley is useful for taxonomy sensitivity and cross-manager dispersion. ([Morgan Stanley, *Mapping Software Exposure in Leveraged Credit*, pp. 4, 17–18](research/Morgan_Stanley-Global_Credit_and_CLO_Strategy_Mapping_Software_Exposure_in_Leveraged_Credit-2026-02-09.pdf))

Accordingly, the current decision-useful view is **about 8% direct positive/buildout exposure plus a separate high-teens to mid-20s software vulnerability screen**. Reporting 27–34% by adding the two would be wrong because the definitions, universes and risk signs differ, and some AI/cloud borrowers can sit in both groups.

## 2. How should it be measured?

### Recommended method

The market estimate should retain the BIS’s matched loan-stock method. A portfolio or mandate should improve on it with loan-level look-through:

\[
C^{+}=\frac{\sum_i EAD_i b_i}{\sum_i EAD_i},\qquad
C^{-}=\frac{\sum_i EAD_i d_i}{\sum_i EAD_i}
\]

where `EAD` is funded principal plus an explicitly modelled expected draw on unfunded commitments; `b` is the fraction of exposure attributable to AI buildout; and `d` is the fraction economically vulnerable to AI disruption. Scores range from zero to one and must be supported by use of proceeds, borrower revenue, collateral, tenant/guarantor, power availability and refinancing dependence. Unknown exposure should remain “unclassified,” not be set to zero. Fair-value and NAV versions should be reported as secondary risk views, rather than substituted for gross EAD.

The main denominator should be **gross credit EAD**, because NAV is distorted by vehicle leverage. The same exposures should also be shown as a percentage of NAV to measure investor loss amplification; the IMF notes that software exposure can exceed 50% of NAV equity in leveraged private-credit vehicles even when it is roughly one-fifth of direct loans. ([IMF, *Global Financial Stability Report*, April 2026, Chapter 1, p. 20](https://www.imf.org/-/media/files/publications/gfsr/2026/april/english/ch1.pdf))

Four controls are necessary:

- **Separate stock from flow.** Track standing AI EAD and new AI originations independently.
- **Roll flows into stock explicitly:** `AI stock(t+1) = AI stock(t) + originations − repayments − defaults − sales/securitisations`; then divide by total credit stock at `t+1`.
- **Measure concentration inside the sleeve.** Report top-five/top-ten shares and `HHI = Σs²` / `effective N = 1 ÷ HHI` by borrower, ultimate hyperscaler or tenant, sponsor, manager, campus and power market. A diversified list of data-centre SPVs can still depend on one tenant and one capex budget.
- **Preserve the two signs.** Report buildout and disruption sleeves side by side; do not net them and do not add them without loan-level overlap control.

### Reconciling the conflicting flow estimates

The BIS estimates that AI-related originations exceeded **$40bn in 2025 but were only about 4% of total direct-loan originations**. The OECD reports that AI-focused companies were **34% of recorded private-credit deal value in 2025**, up from 9% in 2024, while total recorded deal value rose from $97bn to $174bn. ([BIS Bulletin 120](https://www.bis.org/publ/bisbull120.pdf); [OECD, *Global Debt Report 2026*, Chapter 2](https://www.oecd.org/en/publications/global-debt-report-2026_e9d80efd-en/full-report/corporate-debt-market-outlook-in-a-transforming-world_cf86a220.html))

These are not rival estimates of the same statistic. The OECD result implies about **$59bn** of AI deal value (`34% × $174bn`), directionally close to the BIS’s “more than $40bn” numerator. The large percentage difference is mainly in the denominator and coverage: Preqin’s disclosed deal-value sample is much smaller and dominated by large transactions, while PitchBook’s direct-loan origination universe is broader; the taxonomies also differ. Therefore **4–34% is not a plausible range for current concentration**. It is evidence that AI dominated a coverage-limited slice of 2025 deal flow, while still representing a much smaller share of broader originations and about 8% of standing stock.

Reported market-size estimates—roughly $1.5–2tn at end-2024 in the FSB review, $2.1tn in Preqin’s 2024 series and more than $2.2tn in the BIS discussion—also use different dates and include different combinations of funds, BDCs and loan types. ([FSB, *Vulnerabilities in Private Credit*, May 2026](https://www.fsb.org/uploads/P060526.pdf); [Preqin, *Private Markets in 2030*, pp. 17–22](research/Preqin-Private_Markets_in_2030-2025-10-10.pdf)) The 8% stock estimate should therefore not be recomputed with an unrelated headline AUM denominator.

The **15% / 26% / 35% private-credit dependency cases** in the prior cross-asset output were scenario scores applied to an illustrative portfolio, not loan-stock measurements; the associated 17–32% total-fund exposure and 64% share of stress loss were also model illustrations. They are excluded from the empirical Private Credit estimate. ([Prior cross-asset output, table in §7](ai_reponse/cross-asset-ai-concentration-v2.pplx.md); [review of prior AI outputs](AI_Cross_Asset_Concentration_Existing_Report_Findings.md))

## 3. What could concentration look like in 2030?

The cleanest forward bridge combines two source forecasts:

- **Numerator:** the BIS projects **$300–600bn** of outstanding private-credit loans to AI firms by 2030, versus more than $200bn today.
- **Denominator:** Preqin projects private-credit AUM rising from **$2.1tn in 2024 to $4.5tn in 2030**, using closed-end fund data plus a top-down BDC estimate.

Preqin separately forecasts about **$1.7tn of direct-lending AUM** in 2030, but that narrower strategy bucket is not a valid denominator for the BIS numerator: it is below the roughly $2.5tn current direct-loan denominator implied by the BIS’s own $200bn/8% observations, revealing a material coverage mismatch. Using $1.7tn would produce a spurious 18–35% result. The broader $4.5tn AUM series is therefore used only as a transparent market-growth normalizer.

| 2030 scenario | AI-related loan stock | Private-credit AUM | Reproducible calculation | Normalized concentration |
|---|---:|---:|---:|---:|
| Slow / capex- or power-constrained | $300bn | $4.5tn | 300 / 4,500 | **6.7% (~7%)** |
| Midpoint | $450bn | $4.5tn | 450 / 4,500 | **10.0%** |
| Fast / externally financed buildout | $600bn | $4.5tn | 600 / 4,500 | **13.3% (~13%)** |

The **7–13% range and 10% midpoint are calculated estimates**, not directly published forecasts. They imply AI-linked loan dollars rise by at least 50% and potentially about 200% from the current “more than $200bn,” while total private-credit AUM more than doubles. That is why the low case can show a lower portfolio share than today even as dollar exposure rises.

Two larger financing estimates support the direction but should not replace this stock calculation:

- Morgan Stanley estimates **$2.9tn of data-centre spending in 2025–28** and derives an **$800bn private-credit financing requirement** after expected cash flow, public debt and securitisation, applying an approximately 70/30 debt-versus-other-capital split to a residual funding gap. It is a cumulative gross financing opportunity, heavily oriented to asset-based finance—not a year-end loan stock. Repayments, refinancing and migration into ABS/CMBS mean it cannot be added to today’s $200bn. ([Morgan Stanley, *Bridging a $1.5tn Data Center Financing Gap*, pp. 6–9, 34](research/Morgan_Stanley-Global_Fixed_Income_Tech_Diffusion_and_GenAI_Bridging_a_1_5tr_Data_Center_Financing_Gap-2025-07-16.pdf))
- J.P. Morgan’s **$5.5tn AI-capex forecast through 2030**, including $4.1tn of debt and up to a **$1.4tn call on private credit/alternative capital**, is broader still: the last bucket includes infrastructure equity, insurers and other private capital as well as credit. It is a capacity envelope, not a private-credit numerator. ([J.P. Morgan, *AI Capex 2.0*, pp. 1–3, 47–49](research/JP_Morgan-AI_Capex_2_0_If_You_Build_It_They_Will_Finance_It_Updated_Forecasts-2026-06-17.pdf))

The 7–13% range is therefore the appropriate central view; the sell-side funding envelopes are an upside warning that exposure could migrate into private markets faster than the narrow borrower taxonomy captures.

## 4. Evidence, assumptions and remaining gaps

The estimate should be used with the following qualifications:

- **Numerator coverage is narrow.** PitchBook borrower verticals identify AI-native and cloud firms but can miss AI-linked data-centre, power, equipment, real-estate and diversified-corporate loans. A use-of-proceeds and ultimate-tenant map would likely produce a higher buildout numerator.
- **The 2030 numerator and denominator are not perfectly aligned.** The BIS numerator is outstanding loan volume, while Preqin AUM includes undeployed capital as well as invested assets. Preqin’s $4.5tn forecast also excludes parts of the open-ended universe, investment-grade private credit, insurer separately managed accounts and private asset-based finance—precisely where some AI infrastructure financing may sit. No source provides a matched 2030 total-direct-loan-stock denominator, so the 7–13% result is a rounded normalization, not a precise forecast.
- **Software is a screen, not a loss estimate.** AI can benefit as well as disrupt software borrowers. The 19–26% range must be decomposed by product substitutability, recurring-revenue quality, customer retention, leverage, cash interest coverage, maturity wall and sponsor support before stress losses are assigned.
- **BDC transparency creates sample bias.** BDCs offer quarterly fair-value and loan-level disclosure, but public BDCs, private BDCs, private-credit CLOs, infrastructure debt and insurer-managed accounts have different portfolios. BDC observations are the best transparent cross-check, not a complete market benchmark.
- **Common counterparties are undercounted.** Borrower/SPV names should be consolidated to the ultimate tenant, guarantor and hyperscaler. The BIS evidence that the top ten BDCs hold more than half of BDC software lending and that borrowers commonly appear across seven or more BDCs shows why manager-level diversification can be illusory.
- **Flow-to-stock retention is unknown.** The future share depends on amortisation, take-outs into public IG, ABS or CMBS, project cancellations, defaults and asset sales. Gross financing forecasts should enter the roll-forward only after those transitions are modelled.
- **Valuation and liquidity amplify uncertainty.** Stale marks, PIK income, amendments and semi-liquid redemption terms can delay recognition. The IMF’s finding that selected perpetual non-traded BDC liquidity buffers can be exhausted within five to seven quarters under a moderate shock is a loss-amplification warning, not an AI-concentration estimate.

**Bottom line:** report **~8% current direct positive AI financing**, a **separate 19–26% software-disruption screen**, and a **~7–13% 2030 positive-stock scenario with a 10% midpoint**. The dashboard should additionally show origination flow, unclassified exposure, top counterparties/tenants, HHI, maturity/refinancing exposure and the same risks as a percentage of NAV. This preserves the difference between observed stock, leading flow, calculated scenarios and judgement—and avoids the false precision of a single blended “AI exposure” number.
