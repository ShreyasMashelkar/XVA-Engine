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
| 2 | **F3 · Exposure Analytics** ⭐ | "Core engine: **Hull-White 1F Monte Carlo, 10k paths, exact sim.** Reprice on every path → **EE / EPE / PFE / ENE / EEPE.** `a` and `σ` are **calibrated on RBI MIBOR history**, not hardcoded." *(linger on PFE vs EPE chart)* |
| 3 | **F3 · Collateral & Margin** | "Apply the **CSA**: threshold, MTA, exact **MPoR close-out** — the default-to-liquidation gap. Uncollateralised vs CCP." |
| 4 | **F3 · XVA Explain** | "**CVA/DVA** from bootstrapped hazard rates off Indian CDS. **FVA** pathwise (no Jensen bias). **MVA** from ISDA SIMM. **KVA** from SA-CCR. **WWR** 3 ways: Cox intensity, Gaussian copula, regime-switching." |
| 5 | **F5 · FRTB-CVA / BA-CVA** | "Regulatory capital: SA-CCR, BA-CVA, FRTB-CVA, RAROC." |

**ONE wow page (pick one):**
- **F9 · AAD Greeks** — "All sensitivities in ~one pricing pass, not bumping each input."
- **F10 · Hybrid XVA** — "Cross-asset netting: INR rates + NSE equity in one netting set."

---

## REAL NUMBERS (₹ Crore — EOD 2026-06-05)

**Book totals (7 counterparties):**
Total XVA **₹5.71 Cr** · CVA ₹2.36 · DVA ₹0.48 · FVA **−₹1.18** (net funding *benefit*) · KVA ₹4.53 · EAD ₹320.6 · RWA ₹74.7 · Capital ₹7.84

**HERO — SBI (lead with this):** AAA, 50bps → **EPE ₹51.0 Cr · PFE95(1Y) ₹52.8 Cr · CVA ₹1.33 Cr** · KVA ₹3.20. *Largest exposure in the book.*
**ICICI:** AA, 80bps → EPE ₹27.7 Cr · PFE95(1Y) ₹28.8 · CVA ₹0.93.
**Insight line:** "HDFC & TATA show **zero EPE but positive DVA/FVA** — out-of-the-money netting sets: no exposure, but a funding/own-credit benefit. That asymmetry is what XVA captures."

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
