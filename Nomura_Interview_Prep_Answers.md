# XVA Engine — Interview Answer Bank (Nomura Quant)

> Model answers grounded in this repo's actual implementation. Each answer gives the
> formula, the "why," and the subtle point that signals depth. ⭐ = high-priority.
> The five recurring ideas: **Jensen/convexity · mean reversion · measure & martingales ·
> variance reduction · correlation/netting aggregation.** Master those and most of this collapses.

---

## 0. Big-picture / architecture

**⭐ 60-second overview.**
"It's an end-to-end counterparty-credit-risk and XVA engine for INR OTC derivatives. The pipeline is: ingest free market data (FIMMDA OIS, RBI DBIE MIBOR) → bootstrap dual curves → price trades (swaps, swaptions via SABR, equity options via BSM/Heston) → simulate the netting set forward under Hull-White 1F for rates and correlated GBM for equity (10,000 paths) → build the EE/EPE/PFE exposure profiles → compute the full XVA suite (CVA, DVA, pathwise FVA, MVA, KVA) → roll up regulatory capital (SA-CCR EAD, FRTB SA-CVA, BA-CVA) → and wrap it in a pre-trade workflow that prices incremental XVA, checks EAD limits, and tests RAROC against a hurdle. It's pure NumPy/SciPy, validated by a 29-module test suite with martingale and analytical-benchmark checks."

**⭐ Why build it / what problem.**
"A derivatives desk can't quote a price without knowing the *all-in* cost: not just the risk-neutral MTM, but the cost of counterparty default (CVA), funding the uncollateralised position (FVA), posting initial margin (MVA), and holding regulatory capital (KVA). My engine computes that all-in cost and the pre-trade approval economics in one place — that's exactly what an XVA desk does."

**Why INR / what's hard.**
"INR is genuinely harder than USD because the market data is thin and not freely published — there's no free INR swaption vol surface, term OIS/MIBOR basis swaps aren't quoted via API. So I had to anchor a synthetic SABR surface to realised vol from RBI DBIE and parameterise the basis term structure historically. I'm explicit about what's real vs proxied — that data-honesty is itself part of doing this in an emerging market."

**Hardest part / least confident.**
"Hardest: getting the MPoR collateral mechanics resolution-independent — naive integer-step lag gives you a different answer depending on your simulation grid, which is wrong. Least confident: calibrating the wrong-way-risk correlation ρ, because there's no clean market instrument to imply it from — in practice it's a stress parameter, and I treat it as one."

**Why NumPy not QuantLib.**
"I wanted to *own* every formula for learning — bootstrapping, the HW1F bond reconstitution, the LSM regression. QuantLib would have hidden the parts I most wanted to understand. For production you'd absolutely lean on QuantLib/ORE for the validated pricers; I list it because I've used it, but the engine core is deliberately from scratch."

---

## 1. Curves & bootstrapping

**⭐ OIS bootstrapping walkthrough.**
"You have a set of OIS par rates by tenor. Bootstrapping solves sequentially for the discount factors that reprice each instrument at par. For a 1Y OIS, par rate = (1 − DF(1))/annuity, so DF(1) falls straight out. Moving to 2Y, you use the already-solved 1Y DF and solve for DF(2) that makes the 2Y swap par. You march out the curve tenor by tenor — each new instrument adds exactly one unknown DF. Between nodes I interpolate; I check the result reprices inputs to <25 bps and produces no negative forwards."

**⭐ Dual/multi-curve & why post-2008.**
"Pre-2008 one LIBOR curve did both discounting and forward projection. The crisis blew out the basis between collateralised funding (OIS) and unsecured term lending (LIBOR/MIBOR) — they stopped being the same risk. So now you discount collateralised cashflows at the CSA rate (OIS) and project forwards off the term index curve (MIBOR). My `MultiCurveFramework` discounts strictly off OIS and projects strictly off MIBOR, with a basis between them."

**Why discount OIS but project MIBOR.**
"Discounting answers 'what's a future CSA-collateralised cashflow worth today' → that's the collateral remuneration rate, OIS. Projection answers 'what will the floating coupon fix at' → that's the term index, MIBOR. Using one curve for both forces basis = 0, which misprices every floating leg."

**OIS–MIBOR basis economically.**
"It's the credit + liquidity premium of unsecured term lending over (near) risk-free overnight collateralised funding. Term MIBOR embeds rollover/credit risk that overnight OIS doesn't."

**Interpolation choice.**
"I interpolate on the curve such that forward rates stay well-behaved — log-linear on discount factors gives piecewise-constant forwards (stable, no negative forwards), whereas linear-on-zero-rates can produce sawtooth forwards. The choice matters because exposure simulation differentiates the curve — kinks in forwards become artefacts in the EE profile."

**⭐ How do you know the bootstrap is right.**
"Two checks in CI: (1) re-price the input instruments off the built curve — max error <25 bps; (2) no negative instantaneous forwards, which would be a static arbitrage. Both are automated tests."

**Negative forward = arbitrage red flag.**
"A negative forward means a forward-starting deposit pays you to borrow — you could lock in riskless profit. It usually signals a bad interpolation or inconsistent input quotes, not a real market."

**Credit/survival curve.**
"From CDS or bond spreads I bootstrap piecewise-constant hazard rates h_i, giving survival S(t)=exp(−∫h). The marginal default probability in bucket i is S(t_{i−1})−S(t_i) — that's exactly the weight in my CVA sum."

**h, S, spread relationship.**
"For a flat hazard, S(t)=e^{−ht} and the par CDS spread ≈ h·LGD ⇒ h ≈ spread/LGD. My code sets h = spread/LGD with LGD=1−R."

**Extending the bootstrap.**
"Add instrument-based bootstrapping with central-bank-meeting step dates and turn-of-year effects, and a global solver (rather than sequential) so all instruments are repriced jointly under a smoothness penalty."

---

## 2. Pricing models

**⭐ Vanilla swap PV.**
"Value (to the payer) = float leg − fixed leg = N·[(1 − P(0,T_n)) − K·Σ τ_i P(0,T_i)] under single-curve; under dual-curve the float leg is Σ τ_i·F(T_{i−1},T_i)·P_OIS(0,T_i) with forwards off MIBOR and DFs off OIS. The fixed leg is K times the annuity. Par rate is the K that zeroes it."

**Annuity / DV01.**
"Annuity A = Σ τ_i·DF(T_i) — the PV of 1bp of fixed coupon stream. DV01 = N·A·1bp is the cash change in swap value per 1bp parallel rate move. It's the natural risk unit and the input to my MVA's IM proxy."

**European swaption analytically.**
"Express it as an option on the par swap rate. Under the annuity measure the forward swap rate is a martingale; price with Black (lognormal rate) or Bachelier (normal rate). For rates near zero Bachelier/normal vol is the sane choice — INR rates aren't near zero so either works, but I use the SABR-implied vol as the input."

**⭐ Bermudan swaption & why no closed form.**
"It's exercisable on multiple dates, so its value depends on the optimal stopping policy — there's no closed form because the early-exercise boundary is path/state dependent. You need backward induction with a continuation value, which for a Monte Carlo state is Longstaff-Schwartz."

**⭐ SABR SDE & parameters.**
"dF = α F^β dW₁, dα = ν α dW₂, corr(dW₁,dW₂)=ρ. α is the overall vol level (ATM), β the backbone (how vol moves with the forward — 0 normal, 1 lognormal), ρ the skew (correlation of rate and its vol), ν the vol-of-vol (smile curvature/wings)."

**β intuition.**
"β=1 (lognormal) keeps vol proportional to the rate — good for high positive rates; β=0 (normal) keeps absolute vol constant — essential when rates can be low or negative. β controls the ATM 'backbone' the smile slides along."

**Why SABR for swaptions.**
"Quoted swaption vols smile across strike; a flat Black vol misprices away-from-the-money options and, more importantly, gives wrong deltas. SABR fits the smile and gives stable, market-consistent hedges."

**SABR calibration with no INR surface (honesty).**
"Candidly, INR swaption vols aren't free. I anchor the ATM level to realised vol from RBI DBIE and parameterise β, ρ, ν to a plausible shape — so it's a *synthetic but internally consistent* surface, not a calibrated market one. I'm explicit about that limitation; the machinery is correct and would calibrate to real quotes immediately."

**Hagan expansion breakdown.**
"Hagan's implied-vol formula is an asymptotic expansion; it can produce negative densities (arbitrageable smiles) at very low strikes / long maturities. Fixes: a normal-SABR variant or an arbitrage-free PDE/Antonov solution."

**⭐ Heston SDE & why stochastic vol.**
"dS=(r−q)S dt+√v S dW_S, dv=κ(θ−v)dt+ξ√v dW_v, corr=ρ. A static smile is frozen — it can't tell you how the smile *evolves*, so forward-starting and exposure-through-time get it wrong. Heston makes the smile a consequence of variance dynamics, so forward smiles and vol-of-vol tail risk in exposure come out naturally."

**Feller condition.**
"2κθ ≥ ξ² keeps variance strictly positive (the origin is unattainable). If violated, v can hit zero and you need a careful scheme (full-truncation Euler or QE) to avoid negative variance / bias."

**ρ and skew.**
"ρ<0 (typical for equities — leverage effect) makes vol rise as spot falls, producing a downward (negative) skew. Magnitude of ρ controls skew steepness; ν controls how pronounced the wings are."

**Heston vs SABR.**
"SABR for rates/swaptions (per-expiry smile, fast, market standard for the vol cube). Heston for equities/FX where you want one consistent variance process and good forward-smile and term-structure behaviour."

**BSM with dividends.**
"Under Q, S grows at r−q. C = S e^{−qT}N(d₁) − K e^{−rT}N(d₂), d₁=[ln(S/K)+(r−q+½σ²)T]/(σ√T), d₂=d₁−σ√T. The e^{−qT} discounts the spot for foregone dividends."

**Greeks for exposure.**
"Delta and Gamma drive the option's MTM sensitivity along simulated spot paths; Vega matters because exposure is sensitive to the vol you simulate under. For exposure I mainly need a fast, correct revaluation (delta/gamma dominate)."

**Implied-vol inversion failure.**
"I invert BSM for σ via Newton/Brent. It fails when the quoted price violates no-arb bounds (below intrinsic) or for deep ITM/OTM where vega→0 and Newton diverges — I fall back to a bracketed Brent solve."

---

## 3. Monte Carlo engine

**⭐ HW1F SDE & mean reversion.**
"dr=(θ(t)−a·r)dt+σ dW. The −a·r term pulls r toward θ(t)/a at speed a — that's mean reversion, and it's what keeps rates from random-walking off to infinity (and keeps long-rate vol finite). θ(t) is calibrated so the model reprices today's entire discount curve exactly."

**⭐ HW1F vs Vasicek.**
"Same dynamics, but Vasicek's mean-reversion level is constant so it can't fit the initial term structure — it'll mis-price today's bonds. HW1F's time-dependent θ(t) is chosen to match P(0,T) for all T exactly. You never want a model that can't reprice today's curve."

**Negative rates — bug or feature.**
"Feature, by design — HW is Gaussian so r can go negative, which is realistic (EUR/JPY did, and it's the price of analytic tractability). If you must enforce positivity you'd switch to CIR or a shifted-lognormal/Black-Karasinski model, trading tractability for it."

**Analytic ZCB under HW1F.**
"P(t,T)=A(t,T)exp(−B(t,T)·x(t)) with B(t,T)=(1−e^{−a(T−t)})/a and A chosen to match the initial curve and the HW variance term. This is the affine structure — it's why I can reprice bonds on every path without nested simulation, which is exactly what makes LSM and exposure cheap."

**⭐ Discretization / bias.**
"For HW I simulate the state x exactly — the transition is Gaussian with known conditional mean and variance, so there's no Euler discretization bias in the rate. Bonds come from the analytic affine formula. For GBM equity I simulate log-S exactly too. So my discretization error is essentially zero; the only error is statistical (path count)."

**Why 10,000 paths.**
"It's a balance of standard error vs runtime. MC error ~ σ/√N, so 10k gives ~1% relative error on smooth quantities like EPE; tail quantiles (PFE 97.5%) need more or variance reduction. I quote the standard error, not just the point estimate."

**⭐ Beating O(1/√N).**
"Three levers, all implemented: antithetic variates (mirror each normal draw), quasi-Monte-Carlo with scrambled Sobol (near O(1/N) for smooth integrands), and a Brownian bridge to put the largest variance on the first Sobol dimensions. Together that's the standard production variance-reduction stack."

**⭐ Sobol / low-discrepancy.**
"Pseudo-random points clump and leave gaps; a low-discrepancy (Sobol) sequence fills the unit hypercube far more evenly, so the integration error shrinks like (log N)^d/N — close to O(1/N) — instead of O(1/√N). The catch is it degrades in high effective dimension, which is why the Brownian bridge matters."

**Owen scrambling.**
"Plain Sobol is deterministic, so you can't estimate error from it. Owen (nested) scrambling randomises it while preserving the low-discrepancy property — now you can run several scrambles and get an honest variance estimate, and it often improves accuracy too."

**⭐ Brownian bridge.**
"Instead of building a path step-by-step (each step equal importance), the bridge constructs the endpoint first, then the midpoint, then quarters — so the first few Sobol dimensions carry most of the path's variance. Since Sobol is most uniform in its low dimensions, you spend your 'best' coordinates where they matter most."

**Antithetic variates.**
"For each draw Z, also use −Z. For monotone/odd payoffs the estimator variance drops because the two are negatively correlated. It doesn't help (can even slightly hurt) for symmetric/even payoffs where f(Z)=f(−Z)."

**HW2F — why a second factor.**
"One factor forces all rates to be perfectly correlated — the curve can only shift, not twist. Real curves steepen and flatten; HW2F adds a second driver so short and long rates decorrelate, which matters for products sensitive to curve shape and for realistic forward exposure."

**Correlated factors.**
"I Cholesky-decompose the correlation matrix L (LLᵀ=Σ) and apply it to independent normals: correlated = L·Z. That's how rates and equity share a joint Brownian structure in the hybrid model."

**⭐ Longstaff-Schwartz.**
"Backward induction needs the continuation value E[future payoff | state]. In MC you don't have it analytically, so LSM *regresses* realised discounted future cashflows on basis functions of the current state (here the short rate / bond prices). The fitted regression is your continuation value; exercise where immediate value > continuation. Then propagate the resulting optimal cashflows back to today."

**Basis functions.**
"Low-order polynomials in the state (and the immediate exercise value) — typically up to cubic. Too few → biased policy; too many → overfit noise. You only need them accurate near the exercise boundary, not everywhere."

**⭐ Why naive LSM exposure is biased.**
"Foresight/in-sample bias: if you use the same paths to *fit* the continuation value and to *value*, the exercise decision peeks at path-specific noise, inflating value. The fix is to estimate the policy on one set of paths and evaluate cashflows on independent paths (out-of-sample), and to use the *realised* (not fitted) continuation when accumulating value."

**Early exercise reshaping exposure.**
"A European exposure peaks then decays smoothly. A Bermudan gets truncated wherever paths exercise — exposure collapses to zero on exercised paths, so the EE profile is lower and has a different hump. You can't just scale a European profile; you must propagate the stopping rule path-by-path."

---

## 4. Exposure

**⭐ EE / EPE / ENE / PFE.**
"At horizon t: EE(t)=E[max(V_t,0)] (expected positive exposure that moment); ENE(t)=E[max(−V_t,0)]; EPE = time-average of EE over the profile (the scalar used in capital); PFE(t)=q-quantile of max(V_t,0), e.g. 97.5% — a tail measure for limits."

**⭐ Why max(NPV,0).**
"On counterparty default you only lose if they owe you — positive MTM is at risk; negative MTM you still owe. So exposure = max(V,0), an option payoff on the netting-set value. That option-ness is why CVA is non-zero even at zero expected value (convexity)."

**Current vs potential future exposure.**
"Current exposure = today's max(V,0). PFE = a high quantile of *future* exposure from simulation — what it could plausibly become. Limits are set on PFE; CVA integrates EE."

**PFE quantile.**
"I use a high quantile (95–97.5%). It's a risk-appetite choice — higher quantile = more conservative limit. Regulators and IMM backtesting fix the level you must validate against."

**⭐ Hump shape.**
"Two forces fight: diffusion widens the distribution of V over time (exposure ↑), while amortization/roll-down means fewer remaining cashflows as the swap ages (exposure ↓). Early on diffusion wins, late the runoff wins — so EE rises then falls, peaking around 1/3–1/2 of tenor."

**Netting.**
"Within an ISDA netting set you net MTMs before taking max(·,0): exposure = max(Σ V_i, 0) ≤ Σ max(V_i,0). Offsetting trades cancel, so netted exposure is never larger than the gross sum — that's the whole economic value of an ISDA Master Agreement."

**Exposure cube.**
"I persist NPV per (path, time, trade) in Parquet. That lets me net arbitrary sub-portfolios, retrieve trade-level EE, and — crucially — compute *incremental* XVA for a new trade by re-aggregating cached paths instead of re-simulating the book. It's the reuse layer behind the pre-trade workflow."

**Netting across trades per node.**
"At each (path, time) I sum trade NPVs in the netting set, then apply max(·,0). Collateral is subtracted at the netting-set level before the max, with the MPoR lag."

---

## 5. CVA / DVA

**⭐ CVA formula.**
"CVA = LGD · Σ_i EE(t_i) · [S(t_{i−1})−S(t_i)] · DF(t_i). LGD=1−R is loss given default, EE the discounted expected exposure at t_i, [S(t_{i−1})−S(t_i)] the marginal default probability in that bucket, DF the discount factor. It's the expected discounted loss from counterparty default, summed over the life."

**⭐ Why CVA>0 at zero value.**
"Because exposure is max(V,0), a convex function. Even if E[V]=0, E[max(V,0)] = ½E|V| > 0 — Jensen's inequality. The counterparty's optionality on your positive MTM has value regardless of the swap's fair value being zero."

**Unilateral vs bilateral.**
"Unilateral CVA only charges for the counterparty defaulting. Bilateral nets your own default benefit: BCVA = CVA − DVA. It's symmetric — the price two counterparties agree on should be consistent from both sides."

**⭐ DVA & controversy.**
"DVA = LGD_own · Σ ENE(t_i)·marginal-own-PD·DF — the mirror image, a *gain* from your own possible default (you'd not pay the negative MTM). It's controversial because you can't hedge your own default, and you can only realise it by actually defaulting — so booking it as profit is perverse. Basel removes own-credit from regulatory capital for this reason; IFRS 13 still books it."

**DVA perverse incentive.**
"As your credit deteriorates, your DVA rises → you book accounting profit precisely as you approach default. Banks reported large DVA gains in 2011 for exactly this reason — economically hollow."

**Independence assumption.**
"Standard CVA assumes exposure and default time are independent, so you can factor EE × marginal-PD. That breaks under wrong-way risk, where high exposure and default cluster — then you need the joint model (my Cox-process WWR)."

**Recovery/LGD sensitivity.**
"CVA scales ~linearly with LGD, but LGD also enters the hazard via h=spread/LGD, so the spread-implied PD partly offsets — the net recovery sensitivity is smaller than it looks. My AAD engine computes the exact recovery sensitivity in one sweep."

**Stochastic correlated hazard.**
"Then CVA = E[LGD·EE(τ)·DF(τ)·1{τ≤T}] with τ from a correlated intensity — you can't separate the expectations. Positive correlation raises CVA (WWR). That's exactly what the Cox model quantifies."

---

## 6. FVA (your strongest — deep answers)

**⭐ What is FVA.**
"FVA is the PV of the funding cost/benefit of an uncollateralised (or imperfectly collateralised) derivative. When you can't fund a positive MTM at the risk-free rate, you borrow at your funding spread over OIS — that excess is a real cost CVA/DVA don't capture. FVA = FCA − FBA."

**⭐ FCA / FBA / sign conventions.**
"FCA (cost) = funding spread × PV of expected positive exposure you must finance. FBA (benefit) = funding saved when the position is negative and the counterparty effectively funds you. My `fva.py` uses FCA<0, FBA>0, FVA=FCA+FBA; `fva_v2.py` uses both as positive magnitudes, FVA=FCA−FBA. Same net number, two industry sign conventions — I documented both to avoid confusion."

**⭐ Pathwise FVA & the Jensen bias (the killer answer).**
"If borrow and lend spreads are equal, FVA = spread·∫E[V_t]dt — linear in V, so applying the spread to the *average* exposure profile is exact. But my spreads are asymmetric — 150 bps to borrow, 50 bps to lend. That makes the per-path funding cost a *non-linear* (kinked/asymmetric) function f(V): you pay 150 on positive V, earn 50 on negative V. By Jensen, E[f(V)] ≠ f(E[V]) for non-linear f. The EPE-profile approach computes f(E[V]) and is biased; my `compute_fva_pathwise` applies f to each path's NPV then averages — E[f(V)] — which is the unbiased value. The bias is exactly the funding analogue of why CVA>0 at zero value."

**When pathwise = profile.**
"Symmetric spreads. Then f is linear, Jensen is an equality, and the cheap profile method is exact — no need to pay for pathwise. The whole point is that asymmetry is what creates the bias."

**⭐ FVA/DVA overlap (the Hull-White debate).**
"There's a genuine theoretical overlap: FBA (funding benefit on negative MTM) and DVA both stem from your own credit/default, so booking both can double-count. Hull & White (2012) argued FVA shouldn't exist as a separate value adjustment under a no-arb argument; Burgard-Kjær's semi-replication shows FVA and DVA are entangled and you must be careful which 'own-credit' you're counting. Practically, desks define FVA on the funding spread net of own-default to avoid the overlap — I keep FCA/FBA on the funding curve and treat DVA separately, conscious of the seam."

**Funding curve.**
"Funding rate = OIS + funding spread(entity). I discount/accrue the funding leg on OIS and apply the spread to the financed exposure — the spread is the bank's term funding cost over the collateral rate."

**Survival weighting.**
"Funding cost only accrues while both parties are alive, so I weight the path funding integral by bank and counterparty survival probabilities S(t) — no funding cost after default."

**FVA controversial under IFRS 13.**
"FVA is entity-specific (your funding level), which sits awkwardly with IFRS 13's 'exit price / market participant' notion of fair value. The market converged on booking it anyway (JPM took a ~$1.5bn FVA charge in 2014), but it's a Level-3, judgement-heavy reserve."

---

## 7. MVA

**⭐ What is MVA & why it appeared.**
"MVA is the PV of funding the *Initial Margin* you must post over a trade's life. It became material after the BCBS/IOSCO uncleared-margin rules (phased in from 2016) forced bilateral IM — that IM is funded at a spread, and over a long-dated trade the funding cost is significant. MVA = funding spread × ∫E[IM(t)]·DF(t)dt."

**⭐ IM profile estimate.**
"I use a DV01-based SIMM proxy: IM(t) ≈ |DV01|·vol_bps·√(MPoR/252)·z, with z=2.326 for 99%. That's the trade's 1-day P&L vol scaled to the 10-day MPoR and the 99% quantile. I then shape IM(t) over time by the EE profile normalised to its peak, capturing the hump."

**Dynamic IM & why hard.**
"Forward IM depends on the SIMM sensitivities *in the future*, which themselves depend on the simulated state — strictly it's a nested/inner simulation (IM-on-a-path), which is expensive. The DV01×vol proxy is the standard cheap approximation that avoids the nested MC."

**Why z=2.326.**
"SIMM targets a 99% one-tailed confidence over a 10-day MPoR. z=Φ⁻¹(0.99)=2.326 converts the exposure vol to that quantile."

**MVA vs FVA.**
"FVA funds the variation margin / uncollateralised MTM (which can be positive or negative). MVA funds Initial Margin, which is always posted (one-directional, never returned early) and sized to a tail — so it's a pure, persistent funding drag with no offsetting benefit."

---

## 8. KVA

**⭐ KVA formula & why capital is a cost.**
"KVA = cost-of-capital × ∫E[K(t)]·DF(t)dt, where K(t) is the regulatory capital the trade consumes at time t. Capital is a cost because shareholders demand a return (cost of equity) on capital tied up against the trade — if the trade forces you to hold capital for 10 years, that's a real lifetime cost that belongs in the price."

**Chain.**
"trade → SA-CCR EAD → RWA = 12.5×capital or EAD×risk-weight → capital = ratio×RWA → KVA = CoC × PV of that capital profile. My KVA engine literally calls the SA-CCR module to get EAD, then rolls forward."

**CoC = 12%.**
"I use ~12% as an Indian-bank RoE target net of debt cost. KVA scales linearly in it, so it's a key assumption — a desk would use its own hurdle. I'd quote KVA's sensitivity to CoC."

**KVA double-counting debate.**
"Critics argue charging KVA *and* targeting RoE on the same capital double-counts the return. The counter (Green-Kenyon) is that KVA is the price that *delivers* the RoE — it's not extra. It's the least settled XVA theoretically."

**Why long-dated trades.**
"Capital is held every year until maturity, so KVA integrates over the whole tenor — a 30Y trade carries capital ~6× longer than a 5Y, making KVA disproportionately punishing for long-dated, low-margin business."

---

## 9. Wrong-Way Risk

**⭐ WWR vs RWR.**
"Wrong-way risk: exposure rises in the same states where the counterparty is more likely to default — they're positively correlated, so realised loss exceeds independent-CVA. Classic example: an oil producer selling you oil forwards — if oil crashes, your in-the-money exposure to them spikes exactly as their creditworthiness collapses. Right-way risk is the opposite (exposure falls when they're more likely to default), which *reduces* CVA."

**⭐ Cox-process model walkthrough.**
"The counterparty's default intensity λ(t) follows a CIR process correlated with the Hull-White rate factor that drives exposure. I simulate joint (r, λ) paths. Default time τ is the first time the *integrated* intensity ∫₀ᵗλ ds exceeds an independent unit-exponential draw — the Cox / doubly-stochastic construction. Then CVA = E[LGD·EE(τ)·DF(τ)·1{τ≤T}]. With ρ>0, paths with high exposure also have high λ, so they default earlier and at worse times — CVA rises."

**⭐ Why CIR not Vasicek for intensity.**
"A hazard rate must be non-negative — a negative default intensity is meaningless. CIR's √λ diffusion vanishes at zero and (under Feller) keeps λ>0. Vasicek/Gaussian intensity can go negative, so it's wrong for a hazard."

**CIR SDE.**
"dλ=κ(θ−λ)dt+ξ√λ dW. κ reversion speed, θ long-run hazard, ξ vol-of-intensity. √λ makes vol shrink near zero, giving non-negativity."

**How ρ produces WWR mechanically.**
"r and λ share correlated Brownians (corr(dW_r,dW_λ)=ρ). High-rate states → high swap exposure for a payer; if ρ>0 those states also have elevated λ → higher conditional default probability exactly where exposure is large. The expectation E[EE(τ)] no longer factorises, and it's larger than the independent case."

**⭐ WWR multiplier.**
"I report CVA(ρ)/CVA(ρ=0). It isolates the WWR effect from the *dynamics*, not a hand-waved bump — ρ=0 recovers standard independent CVA, ρ>0 gives the multiplier (often 1.1–1.5× depending on correlation)."

**Copula vs stochastic intensity.**
"My Gaussian-copula WWR ties the default time to the exposure via a copula on the quantiles — simpler, static, easy to calibrate a single correlation, but it's a snapshot dependence with no real-time dynamics and inherits the Gaussian copula's thin tails. The stochastic-intensity model is dynamically consistent (default and exposure evolve under one correlated SDE system) and richer, at higher computational cost. Copula for quick stress, Cox for the rigorous number."

**Integrated-intensity = hazard equivalence.**
"Given the intensity path, default is the first jump of a Poisson process with that rate — equivalently, the first time ∫λ exceeds an Exp(1) threshold. Conditioning on the λ path, P(τ>t)=exp(−∫λ); averaging over λ paths gives the (now stochastic) survival. That's the doubly-stochastic / Cox definition."

**Calibrating ρ.**
"Honestly hard — there's no liquid instrument that implies exposure-default correlation. In practice it's a stress/scenario parameter or estimated from historical co-movement of the counterparty's CDS and the relevant market factor. I treat it as a stress input and report sensitivity rather than claiming a market-implied value."

---

## 10. CSA / Collateral / MPoR

**⭐ CSA effect; Threshold/MTA/IA.**
"A CSA exchanges variation margin to collateralise MTM. Threshold (TH) = unsecured amount before collateral is owed; MTA = minimum transfer amount (don't move tiny sums); Independent Amount/IA = extra cushion (≈ initial margin). Exposure collapses toward zero except for the residual gap during closeout."

**⭐ MPoR & why exposure survives.**
"Margin Period of Risk is the time from the last good margin call to closeout/hedging — typically 10 business days. Even with perfect daily margining, if the counterparty defaults you're unhedged for the MPoR while the market moves, so exposure = the change in MTM over that window. Collateral kills the *level*; MPoR leaves the *gap risk*."

**⭐ Resolution-independent MPoR (your own fix).**
"Naive implementations lag exposure by an integer number of simulation steps, so the answer changes if you switch from weekly to daily grid — wrong. Mine does an exact t−δ lookback (δ = MPoR in years) regardless of grid, and applies a √(dt/δ) diffusion correction so the collateralised exposure's variance matches the true MPoR window. The number is now invariant to simulation resolution, which is what correctness demands."

**Collateral reshaping EE.**
"Collateralised EE is a small, roughly flat residual hump set by the MPoR move size and any threshold — versus the large uncollateralised hump. CSA trades have dramatically lower CVA, which is the point of margining."

**CCP vs bilateral collateral.**
"CCP: daily VM plus IM sized by the CCP's model, with a default fund — MPoR often 5 days. Bilateral under UMR: VM plus SIMM-based IM, MPoR 10 days, segregated IM. CCP generally lower exposure (shorter MPoR, mutualised), but you fund IM (→ MVA)."

**Closeout convention.**
"On default you value the replacement cost — risk-free closeout vs replacement-cost (including the survivor's own CVA/funding) closeout differ. The convention affects the residual exposure and whether DVA/FVA appear in the closeout amount; it's a modelling choice with real CVA impact."

---

## 11. SA-CCR

**⭐ EAD = α(RC+PFE); α=1.4.**
"EAD = 1.4 × (Replacement Cost + PFE add-on). α=1.4 is a supervisory multiplier carried over from the internal-model method, intended to cover model risk, wrong-way risk and the gap between the conservative add-on and a full simulation — it scales the current+potential exposure to a capital-conservative EAD."

**⭐ RC margined vs unmargined.**
"Unmargined: RC=max(V−C,0) — current uncollateralised value. Margined: RC=max(V−C, TH+MTA−NICA, 0) — captures the largest exposure that could build before a margin call, given threshold, MTA and net independent collateral."

**PFE add-on.**
"Add-on = multiplier × Σ hedging-set add-ons, each = supervisory factor × effective notional × supervisory delta × maturity factor. It's a rules-based proxy for potential future exposure — no simulation."

**Supervisory factors for IR.**
"0.5% short-dated, stepping to 1.5% for >5Y in my map — higher SF for longer tenor because potential future moves accumulate with maturity. They're regulator-prescribed, not modelled."

**Supervisory delta.**
"+1 for a long/linear position, −1 short; for options it's a Black-style delta with a supervisory vol, so non-linearity is captured crudely. It signs and scales each trade's add-on within the hedging set."

**Hedging-set netting.**
"Within a hedging set (e.g. same currency IR), add-ons partially offset via prescribed correlation/aggregation; across hedging sets they're added. It gives partial netting recognition without a full covariance simulation."

**SA-CCR vs CEM.**
"CEM (the old Current Exposure Method) ignored netting and margining crudely and used flat add-ons. SA-CCR recognises netting, collateral/margining (the maturity factor reflects MPoR), and direction (deltas) — far more risk-sensitive."

---

## 12. CVA capital — FRTB SA-CVA & BA-CVA

**⭐ Default capital vs CVA capital.**
"SA-CCR/RWA covers losses if the counterparty *actually defaults*. CVA capital covers losses from CVA itself moving — mark-to-market volatility of the CVA reserve as credit spreads and market factors change, even with no default. Two distinct risks: jump-to-default vs spread/MTM migration."

**SA-CVA delta capital.**
"K_delta = √(Σ_b K_b² + Σ_{b≠c} γ_bc·S_b·S_c), where S_b is the bucket-level weighted sensitivity and γ_bc the cross-bucket correlation. It aggregates CVA sensitivities (to credit spreads and rates) like a market-risk charge — same √(quadratic form) structure as FRTB market risk."

**SA-CVA sensitivities.**
"CS01 (CVA sensitivity to counterparty credit spread), IR01 (to rates), and vega if you hedge with options. I compute them by bumping the CVA inputs — and my AAD engine gives the full vector in one sweep, which is the efficient way."

**⭐ BA-CVA reduced.**
"K = √[(ρ·Σ_c SCVA_c)² + (1−ρ²)·Σ_c SCVA_c²], ρ=0.5. The ρ term is the *systematic* component (counterparties move together), the (1−ρ²) term the *idiosyncratic* (name-specific) — ρ=0.5 splits CVA risk into common vs specific, so you get partial diversification across counterparties but not full."

**Supervisory discount factor.**
"(1−e^{−0.05M})/(0.05M) is the regulator's fixed annuity-style factor converting the EAD×maturity into an effective discounted exposure for the SCVA — a standardised stand-in for a real discounting/runoff profile."

**SA-CVA vs BA-CVA eligibility.**
"SA-CVA needs supervisory approval and the ability to compute CVA sensitivities (a real CVA engine + sensitivities); banks that can't fall back to BA-CVA, which needs only EAD, maturity and a supervisory risk weight per name. SA-CVA is more risk-sensitive and usually lower capital if you can run it."

---

## 13. SIMM / Initial Margin

**⭐ What is SIMM.**
"ISDA's Standard Initial Margin Model — an industry-standard, transparent way to size bilateral IM under UMR so two parties compute the *same* margin and avoid disputes. It's a sensitivity-based VaR-like model: weighted deltas/vegas/curvature per risk class, aggregated with prescribed correlations to a 99%/10-day number."

**What's missing in mine (honesty).**
"I implement the IR delta risk class for a single currency — the dominant driver for a rates book. Full SIMM has six risk classes (rates, FX, credit-qualifying, credit-non-qualifying, equity, commodity), plus vega and curvature, and a multi-level cross-bucket/cross-class aggregation. My version is the correct core; extending is mechanical, just more buckets and correlation matrices."

**SIMM aggregation.**
"Weighted sensitivity WS_k = RW_k × s_k per tenor bucket; aggregate within a bucket with the intra-bucket correlation matrix (K_b=√(Σ WS² + Σ_{i≠j}ρ_ij WS_i WS_j)), then across buckets with inter-bucket correlations. Same nested √(quadratic form) pattern as SA-CVA/FRTB."

**INR risk weights.**
"INR is a 'regular volatility' currency in SIMM, so it uses the standard (not high-vol, not low-vol) RW table — e.g. ~114 bps at the short end falling to ~34 bps at the long end, reflecting that long-tenor 1bp moves are less volatile per unit DV01."

---

## 14. AAD / Greeks

**⭐ AAD & the speedup.**
"Adjoint Algorithmic Differentiation computes the entire gradient of one output w.r.t. all inputs in a single reverse pass, at ~3–5× the cost of one valuation — *independent* of the number of inputs. Bump-and-revalue costs N+1 valuations for N Greeks. On my 60-node exposure profile that's a ~60× reduction for the exposure-bucket sensitivities alone. For XVA, where you want hundreds of sensitivities, AAD is the only practical method."

**Forward vs reverse mode.**
"Forward mode propagates one input's derivative through the whole graph — cheap when inputs ≪ outputs. Reverse mode propagates the output's adjoint backward to all inputs — cheap when outputs ≪ inputs. CVA is one number with many inputs → reverse mode wins decisively."

**⭐ Own autodiff reverse sweep.**
"My `Var` type records the computation as a DAG: each op stores its inputs and a local-derivative rule. Forward pass computes values; reverse pass seeds the output adjoint at 1 and walks the graph backward, accumulating each node's adjoint via the chain rule (parent_adjoint += local_derivative × child_adjoint). When it reaches the leaves you have ∂CVA/∂(every input)."

**CS01 / IR01 & validation.**
"CS01 = ∂CVA/∂(credit spread), IR01 = ∂CVA/∂(parallel rate). I validate AAD against bump-and-revalue — they match to numerical precision (and AAD has no bump-size/finite-difference noise), which is the standard correctness check."

**Why AAD matters for XVA specifically.**
"XVA risk is high-dimensional — sensitivity to every curve node, every credit spread, every exposure bucket. Bump-and-revalue would need thousands of full Monte Carlo revaluations daily; AAD makes a full risk run feasible overnight. It's why every tier-1 XVA desk uses it."

**Limitation — kinks through MC.**
"AAD through max(V,0) hits a non-differentiable kink, so pathwise derivatives can be biased/discontinuous there. Fixes: smoothing the payoff, the likelihood-ratio/Malliavin method for the discontinuous part, or measuring the kink's measure-zero contribution. It's the known hard edge of pathwise AAD."

---

## 15. RAROC / pre-trade / incremental XVA

**⭐ RAROC.**
"RAROC = (revenue − expected loss − expenses − XVA costs) / allocated economic capital. It's risk-adjusted because the denominator is capital sized to the trade's *risk*, and the numerator already nets expected credit loss and XVA. You approve if RAROC ≥ hurdle (I use 10%)."

**⭐ Incremental XVA.**
"The XVA *change* the new trade adds to the existing netting set — not its standalone XVA. Because exposure nets, a trade that offsets the book can have negative incremental CVA (it *reduces* total risk) even with positive standalone CVA. Pricing the increment is what makes netting-aware, economically correct quoting possible — and the exposure cube makes it cheap (re-aggregate cached paths)."

**Pre-trade approve/reject.**
"The workflow checks three gates: incremental XVA (is the all-in price still profitable), EAD-limit utilisation (does it breach the counterparty limit), and RAROC vs hurdle (does it clear the return bar). All three must pass before execution."

**EAD limit utilisation.**
"Counterparties have credit limits in EAD terms; pre-trade I compute the incremental EAD and check headroom, so you don't discover a breach after booking. It's the credit-risk gate."

**ASRF behind RAROC capital.**
"Basel's Asymptotic Single Risk Factor (Vasicek/Gordy) model — it assumes one systematic factor and an infinitely granular portfolio, which is what lets capital be computed name-by-name as a function of PD, LGD and a supervisory correlation. It's the theoretical basis of the IRB capital I allocate in RAROC."

---

## 16. IFRS 13 / accounting / P&L attribution

**⭐ XVA through IFRS 13 P&L/balance sheet.**
"XVA is a fair-value adjustment, so it sits on the balance sheet as a reserve and its daily change flows through P&L. CVA and FVA *reduce* the asset's fair value (they're costs); DVA is an own-credit *gain*. The reserve moves each day and product control must explain why."

**Fair-value hierarchy.**
"Level 1 = quoted prices; Level 2 = observable inputs (most vanilla derivatives); Level 3 = significant unobservable inputs. XVA is typically Level 2/3 — it depends on model choices, correlation, funding spreads and (for illiquid names) unobservable credit, so the reserve is judgement-heavy and disclosure-sensitive."

**Sign conventions.**
"CVA: liability/contra-asset (reduces value). FVA: contra-asset (cost). DVA: gain (increases value, own credit). My `ifrs13.py` packages exactly this and flags that DVA is booked for accounting but excluded from regulatory capital."

**⭐ P&L attribution / explain.**
"Decompose the day-over-day XVA reserve move into: market moves (curves, spreads, vols), new trades/unwinds, time decay (theta/runoff), and an unexplained residual. A small residual validates the model and risk; a large residual is a red flag the Greeks don't capture the actual revaluation. It's the daily control that proves the engine is consistent."

**DVA booked but excluded from capital.**
"IFRS 13 requires fair value to reflect own credit, so DVA is booked. But Basel removes own-credit gains from CET1 because you can't realise or hedge them without defaulting — counting them as capital would let a deteriorating bank look stronger. The two frameworks deliberately diverge."

---

## 17. Model validation / backtesting

**⭐ How you validate the engine.**
"Layered: (1) martingale test — E[discounted ZCB under Q] equals today's bond price, confirming the simulation is risk-neutral and arbitrage-free; (2) curve reprice — bootstrap reprices inputs to <25 bps, no negative forwards; (3) MC CVA within 15% of a closed-form flat-hazard benchmark; plus the 29-module regression suite. Independent benchmarks, not just 'it runs.'"

**⭐ Exposure backtesting & IMM.**
"To keep IMM approval for CCR capital, you must backtest the simulated exposure distribution against realised portfolio values: how often does realised value breach the model's predicted quantile (e.g. PFE 95%)? Too many breaches ⇒ the model understates risk ⇒ the regulator zones it amber/red and applies a capital multiplier. My module counts quantile breaches and runs the statistics."

**Kupiec POF test.**
"Proportion-of-failures likelihood-ratio test. Null: the breach rate equals the model's stated tail (e.g. 5% for a 95% PFE). LR_POF = −2 ln[ (1−p)^{N−x}p^x / ((1−x/N)^{N−x}(x/N)^x) ] is χ²(1) under the null. Reject ⇒ the model's quantile is miscalibrated. It tests *frequency* of breaches (not clustering — that needs Christoffersen)."

**Basel traffic light.**
"Over a 250-day window: green (few breaches, model fine, multiplier ~3.0), amber (elevated, multiplier rises 3.4–3.85), red (too many, ~4.0 and model rejected). It's the supervisory consequence of the backtest."

**Martingale test — what you assert.**
"Under Q the discounted ZCB is a martingale, so E[D(0,T)] from the simulated short-rate paths must equal the curve's P(0,T) for every T. If it drifts, your drift/measure is wrong. It's the cleanest single check that the MC is risk-neutral."

**Why 15% not 1%.**
"The analytical benchmark deliberately uses a *flat* EE = EPE approximation, whereas the MC uses the true term-structure of EE — so a 10–15% gap is expected and *correct*, not MC error. If they matched to 1% I'd suspect the benchmark was too close to the model to be an independent check. Knowing why the gap exists is the point."

---

## 18. Hybrid / cross-asset / equity

**⭐ What hybrid captures.**
"A single netting set with both an IR swap and an equity-index option, valued under one joint simulation. Asset-by-asset XVA sums two standalone numbers and *misses* (a) cross-asset diversification in the netting (the legs partly offset) and (b) equity-rate wrong-way effects (rates and equity co-move). Netting before max(·,0) across asset classes is the whole point."

**Joint dynamics.**
"Rate factor: dx=−a·x dt+σ_r dW_r (HW1F). Equity: dS=(r−q)S dt+σ_S S dW_S with r=x+f(0,t), corr(dW_r,dW_S)=ρ. The coupling is in the equity drift — the stochastic short rate r from the HW factor *is* the equity's risk-free drift, so a rate move feeds the equity path."

**Netting swap + option.**
"At each (path, time) I value the swap (analytic HW bonds) and the option (BSM on the simulated S) on the *same* path, sum the two NPVs, then take max(·,0). One correlated simulation, netted exposure, then one CVA/DVA/FVA."

**Rate-equity correlation source.**
"Estimated from historical co-movement of Nifty and the rate factor; it's a key input and I report XVA sensitivity to it rather than treating it as known precisely."

**Why adding equity can reduce CVA.**
"If the equity leg's MTM is negatively correlated with the swap's, the combined netting-set exposure has lower variance than either alone — diversification. Netted EE < sum of standalone EE, so total CVA falls. That's the economic value the hybrid engine reveals and a siloed engine hides."

---

## 19. Software / data / engineering

**Market data sources.**
"OIS from FIMMDA/FBIL, overnight MIBOR from RBI DBIE — both free, fetched live. Real: OIS curve, O/N MIBOR. Proxied: term OIS/MIBOR basis (parameterised historically, anchored to the true O/N spread) and the swaption vol surface (synthetic, anchored to realised vol). I'm explicit about each."

**⭐ Real vs proxied (honesty).**
"I never hide the proxies. Free INR markets don't publish a swaption surface or term basis swaps, so those are clearly-labelled synthetic inputs anchored to real free data. Everything downstream is correct given the inputs; with a Bloomberg feed it would calibrate to live quotes unchanged."

**Testing philosophy.**
"29 test modules — every pricer against an analytical benchmark, every XVA against a hand-calc or limit case, plus the validation suite (martingale, backtest, no-arb). I test *properties* (martingale, monotonicity, netting ≤ gross) not just golden numbers, so refactors stay honest."

**Caching / reuse.**
"The exposure cube persists NPV paths in Parquet, so incremental XVA and sub-portfolio netting reuse simulations instead of re-running them; live market fetches are cached at the app layer; the equity-option MTM is vectorised (~600× over the naive loop)."

**Scaling to 1M trades.**
"First bottleneck is memory for the path×time×trade cube. Fixes: chunk by netting set, store float32, push aggregation into a columnar engine (Parquet + DuckDB/Spark), and only keep netting-set-level NPVs not per-trade where possible. Simulation itself parallelises trivially across paths."

**Parallelise the MC / GPU.**
"Paths are independent → embarrassingly parallel: vectorise across paths in NumPy (done), then multiprocess across netting sets, then GPU (CuPy/JAX) for the path generation and revaluation. AAD maps well to GPU too."

---

## 20. Limitations & "is this really yours"

**⭐ Top limitations.**
"(1) Single-currency INR — no FX/XCCY XVA. (2) Simplified SIMM (IR delta only) and reduced FRTB SA-CVA. (3) WWR correlation ρ isn't market-calibratable — it's a stress input. (4) Synthetic swaption surface (no free INR vols). (5) DIM uses a proxy, not nested simulation. I scoped depth-over-breadth deliberately."

**What I simplified.**
"Single global yield-curve model per simulation, proxy IM instead of nested MC, flat recovery, no collateral haircuts/rehypothecation, no central-clearing default-fund capital. None change the architecture — they're parameter/coverage extensions."

**⭐ What next with another month.**
"Add FX and cross-currency swaps with a multi-currency HW + FX-GBM model and the FX-XVA quanto/correlation effects; build true nested-simulation Dynamic IM for MVA; and calibrate the SABR surface to real (paid) quotes. Then a proper FVA/DVA overlap treatment via Burgard-Kjær semi-replication."

**Numbers I trust most/least.**
"Most: the analytically-anchored ones — swap PV, HW bond prices, CVA on a vanilla netted swap (validated). Least: anything driven by an unobservable — WWR multiplier (ρ), MVA (proxy IM), and absolute swaption vols (synthetic surface)."

**Debug 'CVA 20% too high'.**
"Bisect the pipeline against benchmarks: check the curve reprices (bootstrap), the martingale test (simulation drift), the EE profile shape vs analytic, then the credit curve / LGD, then collateral/MPoR. Compare to the closed-form flat-hazard CVA — if MC ≫ benchmark beyond the expected 15%, the simulation or curve is the culprit; if the benchmark itself is high, it's the credit/LGD inputs."

**What I learned beyond textbooks.**
"That the hard part isn't the formulas — it's the *seams*: resolution-independence in MPoR, the FVA/DVA overlap, in-sample bias in LSM, the Jensen bias when funding spreads are asymmetric. Textbooks give the clean integral; building it forces you to confront where the clean integral is subtly wrong."

---

## 21. First-principles math curveballs

**⭐ Itô's lemma → d(log S) under GBM.**
"Itô: df = (∂f/∂t + μ∂f/∂x + ½σ²∂²f/∂x²)dt + σ∂f/∂x dW. For GBM dS=μS dt+σS dW and f=ln S: ∂f/∂S=1/S, ∂²f/∂S²=−1/S². So d ln S = (μ − ½σ²)dt + σ dW. The −½σ² is the Itô correction — it's why the *median* of S drifts below the mean and why lognormal pricing isn't just exp of the drift."

**⭐ Why CVA>0 at E[V]=0.**
"Jensen: exposure max(V,0) is convex, so E[max(V,0)] ≥ max(E[V],0). At E[V]=0, E[max(V,0)]=½E|V|>0. The counterparty holds an option on your positive MTM; options have value even on a zero-mean underlying. Same reason FVA's asymmetric-spread bias exists — it's one idea."

**⭐ P vs Q measure.**
"P is the real-world measure (actual drifts, used for risk/backtesting/PFE quantiles you'll be held to). Q is the risk-neutral measure where discounted tradables are martingales and every asset drifts at the risk-free rate — used for *pricing*, because no-arbitrage prices are Q-expectations. I simulate exposure for *valuation* (CVA/FVA) under Q, but exposure *backtesting* is a P-measure exercise — getting that split right matters."

**Martingale & why discounted prices are martingales under Q.**
"A martingale has E[X_t|F_s]=X_s — no predictable drift. Under Q with the money-market numéraire, the fundamental theorem of asset pricing says discounted tradable prices are martingales — that's equivalent to no-arbitrage. It's the property my validation test checks (E[D(0,T)]=P(0,T))."

**E[max(V,0)] for V normal (Bachelier exposure).**
"If V~N(μ,σ²): E[max(V,0)] = μΦ(μ/σ) + σφ(μ/σ). At μ=0 it's σφ(0)=σ/√(2π)=σ·0.3989. This is the Bachelier call formula — and it's the analytic EE for a normal exposure, which I use as a sanity benchmark."

**Girsanov in one sentence.**
"Changing measure (e.g. P→Q) shifts the Brownian motion's drift by the market price of risk while leaving its volatility unchanged — it's how the real-world drift becomes the risk-free drift for pricing. It shows up implicitly every time I simulate exposure under Q rather than P."

**Brownian motion / quadratic variation / dW²=dt.**
"BM has independent Gaussian increments, continuous paths, and quadratic variation [W]_t = t. Heuristically (dW)²=dt because Var(dW)=dt and the variance of (dW)²−dt is O(dt²)→0, so to first order (dW)² is deterministically dt — that's the engine of the Itô correction term."

**Variance of correlated normals → covariance matrix.**
"Var(Σw_iX_i)=wᵀΣw — cross terms need the full covariance, not just the diagonal. This is *exactly* the √(quadratic form) structure in SIMM, SA-CVA and FRTB aggregation (Σ WS² + Σ_{i≠j}ρ WS_iWS_j) and in netting two correlated exposures — same algebra everywhere."

**ES vs VaR & coherence.**
"VaR is a quantile; Expected Shortfall is the average loss beyond it. ES is *coherent* (sub-additive — diversification never increases it), VaR is not (it can penalise diversification in fat-tailed/discontinuous cases). FRTB moved market-risk capital from 99% VaR to 97.5% ES for exactly this; it's why PFE-style tail measures should be handled carefully."

---

### Closing meta-point for the interviewer
"If there's one thread through the whole engine, it's **Jensen's inequality and convexity** — it's why CVA is positive at zero value, why pathwise FVA differs from profile FVA under asymmetric funding, and why exposure has option-like value at all. And the second thread is **correlation aggregation** — the same √(quadratic form) shows up in netting, SIMM, SA-CVA and FRTB. Once you see those two patterns, the engine stops being twenty modules and becomes two ideas applied carefully."
