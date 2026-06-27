# XVA ENGINE — ZOOM WALKTHROUGH CHEAT SHEET

**Run:** `streamlit run app/streamlit_app.py` → http://localhost:8501  ·  Backup: `demo/XVA_Engine_Demo.webm`
**Pre-call:** start app 5 min early, warm up F3/F4/F5, browser ~90% zoom, silence notifications.

---

## OPENING (no share, 1 min)
> "End-to-end **XVA & counterparty-risk platform for INR OTC interest-rate derivatives** — mirrors a real Rates/CCR/XVA desk, built on free public data (RBI, FIMMDA, FBIL). I'll walk the pipeline: **pricing → exposure → collateral → XVA → capital.** How deep on the math do you want me to go?"

---

## PIPELINE (share screen, ~12 min)

| # | Page to click | Say |
|---|---|---|
| 1 | **F4 · Rates & Vol** | "Starts with the **INR OIS curve** (FBIL MIBOR) + G-Sec. **Dual-curve**: OIS discount, MIBOR projection. Pricer gives MTM, par, **DV01/PV01, key-rate DV01, gamma**. SABR for swaptions." |
| 2 | **F3 · Exposure Analytics** ⭐ | "Core engine: **Hull-White 1F Monte Carlo, exact sim.** Reprice on every path → **EE / EPE / PFE / ENE / EEPE.** `a` and `σ` are **calibrated on RBI MIBOR history**, not hardcoded. This exposure is a **market-risk** measure — same for any counterparty; the name only matters at CVA." *(linger on EE vs PFE — quote live numbers below)* |
| 3 | **F3 · Collateral & Margin** | "Apply the **CSA**: threshold, MTA, exact **MPoR close-out** — the default-to-liquidation gap. Uncollateralised vs CCP." |
| 4 | **F3 · XVA Explain** | "**CVA/DVA** from bootstrapped hazard rates off Indian CDS. **FVA** pathwise (no Jensen bias). **MVA** from ISDA SIMM. **KVA** from SA-CCR. **WWR** 3 ways: Cox intensity, Gaussian copula, regime-switching." |
| 5 | **F5 · FRTB-CVA / BA-CVA** | "Regulatory capital: SA-CCR, BA-CVA, FRTB-CVA, RAROC." |

**ONE wow page (pick one):**
- **F9 · AAD Greeks** — "All sensitivities in ~one pricing pass, not bumping each input."
- **F10 · Hybrid XVA** — "Cross-asset netting: INR rates + NSE equity in one netting set."

---

## REAL NUMBERS

### LIVE — quote THESE while on the F3 page
Inputs on screen: **₹500 Cr notional · 5Y · receive-fixed · 2,000 paths.**
- **EE peaks ~₹12.6 Cr** at inception (t=0), decays linearly to **₹0 at 5Y** as the swap amortises.
- **EE ≈ ₹5.3 Cr** at t=2.5Y (per the MTM-distribution box).
- **EPE (time-avg of EE) ≈ ₹5–6 Cr.**
- Say: *"This ₹500 Cr 5Y receive-fixed swap has EE peaking ~₹12.6 Cr and EPE ~₹5–6 Cr, decaying to zero at maturity. PFE95 tracks just above EE. This exposure is **counterparty-independent** — it's pure market risk."*

### ⭐ MONEY MOVE — exposure vs credit separation (do this live)
Switch the **COUNTERPARTY** dropdown SBI → NBFC_X. *(SBI = AAA, 50bps · NBFC_X = BBB, 300bps)*
Verified figures (same ₹500 Cr trade): **CVA ₹0.125 Cr → ₹0.706 Cr** (EE unchanged at ₹12.6455 Cr).
> "Watch — the **EE curve doesn't move**, because exposure is market risk. But **CVA jumps ~5.6×** (₹0.13 → ₹0.71 Cr), because *that's* where the counterparty's default risk enters. Exposure × probability-of-default = CVA. That separation is the heart of XVA."

*If asked "why 5.6× not 6× when spreads are 6× apart?"* → "CVA isn't linear in spread — a higher hazard rate also lowers survival probability, so there's less time alive to be exposed; that interacts with discounting. The 6× spread ratio gets damped to ~5.6×."

### EOD BATCH — full book (only on the EOD/reporting view, label clearly as "full book")
*Different/larger netting set than the single live trade above — keep them separate.*
- Book totals (7 cps): Total XVA **₹5.71 Cr** · CVA ₹2.36 · DVA ₹0.48 · FVA **−₹1.18** (net funding *benefit*) · KVA ₹4.53 · EAD ₹320.6 · RWA ₹74.7.
- SBI (full netting set): EPE ₹51.0 Cr · PFE95(1Y) ₹52.8 · CVA ₹1.33.
- **Insight line:** "HDFC & TATA show **zero EPE but positive DVA/FVA** — out-of-the-money netting sets: no exposure, but a funding/own-credit benefit. That asymmetry is what XVA captures."

> ⚠️ Two traps to avoid: (1) Don't quote the ₹51 Cr EPE while the live single-trade page (~₹5–6 Cr) is on screen. (2) Don't call the live EE "SBI's exposure" — it's the **trade's** exposure, identical for any counterparty.

---

## HONEST CLOSE (30 sec)
> "One caveat I'm upfront about: NSE/CCIL/FIMMDA feeds are access-restricted, so those inputs are simulated; RBI data is live. All flagged LIVE / CACHED / SYNTHETIC in the UI and documented. Where's most interesting to go deeper?"

---

## HARD-QUESTION FALLBACKS
- **Why Hull-White?** → "Tractable, exact-simulatable, mean-reverting — standard for rates exposure. HW2F page for richer dynamics."
- **How validated?** → **F7** has exposure backtesting + model validation.
- **Data real?** → "RBI/FBIL real; restricted feeds simulated — flagged in UI."
- **Don't know?** → "Good question — let me note it and follow up." *Never bluff a quant.*

---
**10 categories total (F1–F10).** Don't show all of them — follow the 5-step pipeline above.
