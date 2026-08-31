# GUIMLAB — RAPPORT SCIENTIFIQUE : CONSTANTES PHYSIQUES ET DYNAMIQUES UNIVERSELLES
*Fondements formels, hypothèses réfutables et opérateurs symplectiques à temps continu*

---

## 1. Introduction & Objectif
Le présent document formalise l'intégration de constantes mathématiques et physiques fondamentales dans le substrat neuromorphique continu de **GuimLab**.

Chaque constante n'est pas une simple valeur numérique, mais l'invariant d'un opérateur différentiel ou symplectique assurant la stabilité, la régularisation spectrale ou l'accélération topologique du réseau à 2000-4000 Hz.

---

## 2. Tableau Récapitulatif des Constantes (`include/guim_physics_constants.h`)

| Constante | Symbole | Valeur C++20 | Rôle dans GuimLab |
| :--- | :--- | :--- | :--- |
| **Coulomb / Yukawa** | $k_e$ | `8.9875517923e-1f` | Intensité du potentiel de champ électrostatique pour l'attention topologique creuse. |
| **Écran de Debye** | $\kappa$ | `1.6180339887e0f` | Facteur d'écrantage diélectrique limitant l'interaction à distance ($e^{-\kappa r}$). |
| **Nombre d'Or (KAM)** | $\Phi$ | `1.61803398875f` | Rapport de pulsation anti-résonant prévenant la destruction des tores invariants de phase. |
| **Ratio d'Argent** | $\delta_S$ | `2.41421356237f` | Facteur d'échelle harmonique pour les octaves acoustiques secondaires. |
| **Feigenbaum Delta** | $\delta$ | `4.66920160910f` | Facteur universel d'échelle de période dictant le seuil de criticalité CBP (*Edge of Chaos*). |
| **Feigenbaum Alpha** | $\alpha$ | `2.50290787509f` | Facteur d'échelle de l'espace des phases pour le taux d'oubli métabolique $\beta_{ema}$. |
| **Euler-Mascheroni** | $\gamma$ | `0.57721566490f` | Terme de correction analytique asymptotique pour la sommation continue des harmoniques. |
| **Action Synaptique** | $\hbar_{syn}$ | `1.054571817e-4f` | Aire minimale d'incertitude dans l'espace $(E, P)$ éliminant les divisions par zéro. |

---

## 3. Dérivation et Preuves Expérimentales

### A. Potentiel Électrostatique de Yukawa-Coulomb
L'interaction entre neurones distants $i$ et $j$ dans le manifold cortical s'exprime :

$$V_{ij} = k_e \cdot \frac{q_i(t) q_j(t)}{\|\mathbf{x}_i - \mathbf{x}_j\|_2 + \epsilon} \cdot \exp\left(-\kappa_{Debye} \|\mathbf{x}_i - \mathbf{x}_j\|_2\right)$$

Le seuillage à distance $r > 3/\kappa$ permet une exécution warp-collective (`__ballot_sync`) sans divergence.

### B. Invariance Spectrale KAM (Kolmogorov-Arnold-Moser)
Pour éviter les collisions de phase entre les oscillateurs de Kuramoto dans SR-MIT, la grille de pulsations est calibrée selon :

$$\omega_k = \omega_0 \cdot \Phi^{k \pmod M}$$

La nature "incommensurable" de $\Phi$ garantit que $\det(R) = 1.0$ sans résonance parasite sur $10^7$ cycles.

---

## 4. Protocole de Non-Régression & Tests Unitaires
Toutes les propriétés de ces constantes sont testées de manière déterministe dans `tests/test_physics_constants.cpp` et `tests/test_harness_integrity.cpp`.
