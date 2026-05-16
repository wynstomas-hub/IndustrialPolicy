# The Structural Collapse of the European Industrial Leash

**Tomas Wyns**
**[May 2026]**

---

This repository contains the  dataset and the Python econometric code used to generate the key visualisations for the analysis of the post-2022 decoupling of the European industrial ecosystem.

Historically, Europe's foundational chemical industry functioned as a strategic, vertically integrated anchor, cointegrated in statistical lockstep with high-volume domestic manufacturing sectors (Automotive, Plastics/Rubber, and Construction). This project utilises advanced rolling econometric models to mathematically demonstrate the terminal collapse of this relationship, validating the hypothesis of mass import substitution for foundational molecules.

## Visual Portfolio

These visualisations confirm that European downstream demand has permanently decoupled from European upstream supply. The mechanical gravitational pull—the "leash"—that once bound the continent’s industrial matrix has snapped.

### 1. The Macroeconomic Leash Collapse (Heatmap)

**How to Interpret:**
This heatmap uses rolling P-values to establish the strength of the industrial equilibrium ("the leash") over a two-decade window.

* **Top Bar (Johansen Systemic Gravity):** This measures the health of the entire ecosystem (all 4 variables acting together). Historically, the system remained cohesive, weathering the 2008 crash and COVID-19. As the solid red bar post-2022 confirms, the *entire system* has suffered a total structural failure.

* **Bottom Panel (Engle-Granger Pairwise Fraying):** These rows map individual bilateral relationships, using a continuous visual gradient from Dark Green (Strong Coupling) to Dark Red (Catastrophic Decoupling). You can visibly trace the construction "slow bleed" from 2017/2018, the pandemic plastic fracture in 2020, and the universal "wall of fire" decoupling across all links following the 2022 shock.

---

---

## Technical Methodology

The visualisations rely on a rolling application (84-month windows) of standard econometric stationarity and cointegration testing, applied to Eurostat indices of industrial production (1998-2026) for NACE Rev. 2 sectors C20 (Chemicals), C22 (Rubber/Plastics), C29 (Motor Vehicles), and F (Construction). Stationarity was confirmed via Augmented Dickey-Fuller (ADF) tests ensuring an I(1) integration order across the entire panel. Pairwise long-run equilibrium was assessed using Engle-Granger tests (AIC dynamic lag selection), while overarching systemic gravity was measured simultaneously across all variables using the Johansen Multivariate Trace Statistic.

## Repository Instructions

### Structure
* `chemdogleash.py`: The single Python script required to reproduce both visual portfolio items.
* `EU_Chemicals_Academic_Data.csv`: The primary academic dataset containing the foundational index numbers and the calculated rolling P-values.
* `requirements.txt`: List of required Python packages (pandas, matplotlib, seaborn, etc.) needed to run the script.
* `/graphs`: Directory containing the high-resolution PNG outputs used in the publication.

### How to Reproduce these Graphs locally:

1.  **Clone this Repository** using git or by downloading the ZIP file.
2.  **Ensure Python is installed** along with `pip` (the Python package manager).
3.  **Install the necessary dependencies** by running this command in your terminal:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the core script:**
    ```bash
    python chemdogleash.py
    ```
    The script will update the PNG images in the `/graphs` folder.

---

## License & Citation

This project is licensed under the **MIT License**.

If you utilise this data, code, or visual methodology in academic or industrial research, please provide appropriate citation to this repository:

> **Tomas Wyns**, (May 2026), "The Structural Collapse of the European Industrial Leash: Visual Portfolio," GitHub Repository. 
