# CE-CBMN: Co-Evolutionary Common Biological Market Networks 🍄🌐

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Paper Status](https://img.shields.io/badge/Paper-Submitted_to_SciELO-success.svg)](#)

> **Author:** Adan Facundo ([ORCID: 0009-0003-9110-4744](https://orcid.org/0009-0003-9110-4744))  
> **Preprint:** [Link to SciELO / DOI will be updated here]

## 📖 Overview

The **CE-CBMN (Co-Evolutionary Common Biological Market Networks)** algorithm introduces a completely new paradigm in evolutionary computation and the optimization of dynamic and stochastic environments. Traditionally, genetic algorithms suffer an elite population collapse when faced with unexpected environmental shocks, as they rely exclusively on static mathematical fitness functions.

To address this vulnerability, CE-CBMN draws inspiration from the symbiotic interactions and biological markets of real-world mycorrhizal networks, successfully preventing population collapse by prioritizing global ecosystem resilience.

## ✨ Core Innovations

This model distinguishes itself from traditional Genetic Algorithms and Swarm Intelligence models through three fundamental innovations:

1. **Mathematical Decoupling:** It implements a strict mathematical separation between the theoretical potential of a solution ($F_i$) and its operational energy or living biomass ($B_i$).
2. **Dynamic Biological Market:** Fungal edges act as independent agents that regulate resource transfers based strictly on market profitability ($\Pi_j$) and a greed threshold ($\tau_{corte}$). Unprofitable connections are pruned.
3. **Phenotypic Defenses & Topological Alerts:** Features a phenotypic defense mechanism ($D_i$) that is rapidly activated by stress signals (Wave 3) to shield neighboring nodes *before* a localized stochastic shock completely destroys them.

## 🚀 The Simulation (Benchmarking)

The provided Python script simulates the CE-CBMN algorithm operating on a **2D Inverted Rastrigin function**, a highly multimodal topology. 

To test its resilience, a severe localized Gaussian shock is triggered exactly at **Generation 40**, aimed directly at the global optimum coordinate `(0,0)`. While standard algorithms collapse under this shock, CE-CBMN triggers its Wave 3 defense and market redistribution to keep the network alive.

### Running the Code

1. Ensure you have Python and `numpy` installed.
2. Clone this repository:
   ```bash
   git clone [https://github.com/facuyami196IA/CE-CBMN-Algorithm.git](https://github.com/facuyami196IA/CE-CBMN-Algorithm.git)
   cd CE-CBMN-Algorithm

   
## 📝 Citación y Referencias (Citation)

El documento formal de investigación y las bases matemáticas de **CE-CBMN** han sido publicados oficialmente bajo acceso abierto en Zenodo. Si utilizas este algoritmo o su código en tu investigación, por favor cítalo de la siguiente manera:

**DOI Oficial:** [10.5281/zenodo.20129871](https://doi.org/10.5281/zenodo.20129871)

**Formato APA:**
Yamil, F. Y. A. (2026). *CE-CBMN: Co-Evolutionary Common Biological Market Networks - Formal Specification*. Zenodo. https://doi.org/10.5281/zenodo.20129871

**Formato BibTeX:**
```bibtex
@misc{yamil_ce_cbmn_2026,
  author       = {Facundo Yamil Adan Yamil},
  title        = {CE-CBMN: Co-Evolutionary Common Biological Market Networks - Formal Specification},
  month        = {May},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20129871},
  url          = {[https://doi.org/10.5281/zenodo.20129871](https://doi.org/10.5281/zenodo.20129871)}
}
## 📝 Citación y Referencias (Citation)

El documento formal de investigación y las bases matemáticas de **CE-CBMN** han sido publicados oficialmente bajo acceso abierto en Zenodo. Si utilizas este algoritmo o su código en tu investigación, por favor cítalo de la siguiente manera:

**DOI Oficial:** [10.5281/zenodo.20129871](https://doi.org/10.5281/zenodo.20129871)

**Formato APA:**
Yamil, F. Y. A. (2026). *CE-CBMN: Co-Evolutionary Common Biological Market Networks - Formal Specification*. Zenodo. https://doi.org/10.5281/zenodo.20129871

**Formato BibTeX:**
```bibtex
@misc{yamil_ce_cbmn_2026,
  author       = {Facundo Yamil Adan Yamil},
  title        = {CE-CBMN: Co-Evolutionary Common Biological Market Networks - Formal Specification},
  month        = {May},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20129871},
  url          = {[https://doi.org/10.5281/zenodo.20129871](https://doi.org/10.5281/zenodo.20129871)}
}
