# GUIMLAB — CONSTITUTION & DIRECTIVES SYSTÈME

## 1. RÈGLE D'OR : 100% C++ NATIF BARE-MÉTAL & CUDA
Le moteur et le runtime du projet GuimLab ne doivent contenir **aucun code autre que du C++ natif (C++20) et CUDA natif**.

- **Zéro dépendance lourde / framework externe** : Aucun runtime Python en production/inférence, aucun framework externe (zéro PyTorch, zéro LibTorch, zéro cuDNN, zéro ONNX Runtime).
- **Bare-metal pur** : Accès direct aux registres GPU, assembleur PTX, instructions intrinsèques CPU (AVX2/AVX-512), mémoire partagée POSIX/Windows SHM sans copie.
- **Zéro allocation dynamique sur le chemin critique** : Aucune allocation heap (`malloc`, `new`, `std::vector::resize`, `cudaMalloc`) dans les boucles de tick/inférence/apprentissage. Mémoire pré-allouée en arènes statiques et ring buffers sans verrou.

---

## 2. PARADIGME ALGORITHMIQUE : INUSITÉ, FORMEL & INTERPOLABLE
- **Algorithmique fermée & Traces continues (CF-TT)** : Dynamiques d'apprentissage et de traces d'éligibilité dérivées sous forme analytique exacte et interpolable en temps continu (sans discrétisation naïve ni BPTT).
- **Continual Backpropagation (CBP) & Plasticité** : Préservation continue de la plasticité synaptique (Dohare & Sutton 2024), réinitialisation asynchrone des unités mortes sans rupture de l'inférence.
- **Méta-Apprentissage Synaptique (TMD-ET)** : Vecteur de pas d'apprentissage adaptatif par synapse (IDBD méta-gradient direct).
- **Architecture Dual-Speed** :
  - **Reflex L0** : 32×32 fixé dans les registres GPU (~100-200 ns, 0 lecture VRAM).
  - **Cortex L1/L2** : Kernel fusionné, activation creuse DDWR (`__ballot_sync`), modulation Tensor Cores (`mma_sync`), latence budget < 5 μs.

---

## 3. ÉLÉGANCE, RIGUEUR ET RÉSULTAT
- Code compact, auto-documenté par sa structure mathématique.
- Alignement mémoire strict sur les lignes de cache (64 octets, `alignas(64)`).
- Validation systématique par compilation native, benchmarks réels (`ctest`, `benchmarks/`) et mesures au cycle près (`RDTSC`, `cudaEvent`).

---

## 4. INTERDICTIONS ARCHITECTURALES STRICTES & ANTI-PATTERNS PROSCRITS
Tout agent ou LLM intervenant sur ce dépôt DOIT respecter ces interdictions absolues :

1. **PROHIBITION 1 : JAMAIS de CUDA Graph sur les Persistent Kernels**
   - *Raison technique* : `kiss_reflex_persistent_kernel` et `guim_lovelace_fused_kernel` sont des kernels résidents à exécution continue (`while (!terminate)`) synchronisés par mémoire partagée POSIX SHM (`seq_in`/`seq_out`).
   - *Conséquence d'une violation* : `cudaStreamBeginCapture` gèle le driver CUDA sur toute boucle infinie de spin-wait. Le CPU n'émet déjà aucun lancement de kernel par tick en régime établi.

2. **PROHIBITION 2 : JAMAIS de dynamique non-symplectique rompant l'espace $(E_{ij}, P_{ij})$ de SR-MIT**
   - *Raison technique* : Le ratio de convergence ($50\,000\times$ sur Mackey-Glass) et le suivi sans retard des formants reposent sur l'opérateur de rotation unitaire symplectique $\det(R)=1$, l'invariant hamiltonien $H_{ij} \le H_{max}$ et le verrouillage de fréquence de Kuramoto.
   - *Conséquence d'une violation* : Remplacer cette dynamique par des ODE "liquides" (LTC) heuristiques non-hamiltoniennes détruit la stabilité de phase et entraîne la divergence du gradient.

3. **PROHIBITION 3 : JAMAIS de KAN (B-splines / activations par synapse) sur le Reflex L0 et les boucles synchrones**
   - *Raison technique* : Le Reflex L0 est un warp unique de 32 threads (`__launch_bounds__(32, 1)`) avec 0 divergence et 0 octet de spill de registres (`check_l0_spill.sh`).
   - *Conséquence d'une violation* : L'évaluation de nœuds de spline conditionnels introduit une divergence de warp (`warp branch divergence`) et sature le fichier de registres (spill en mémoire locale), faisant exploser la latence réflexe au-delà de 100 ns.

4. **PROHIBITION 4 : JAMAIS de dépendance ou pont synchrone bloquant vers des LLMs discrets (MCP/RPC) dans le Hot-Loop**
   - *Raison technique* : GuimLab est un système d'exploitation neuromorphique continu opérant à 2–4 kHz ($308.5\ \mu\text{s}$ p50). Tout couplage externe doit être strictement asynchrone (ring-buffer sans verrou) et isolé de la boucle acoustique.

5. **PROHIBITION 5 : INTÉGRITÉ DU CODE FONCTIONNEL**
   - Aucun agent ne doit modifier le code C++/CUDA stabilisé sans validation préalable par le protocole de preuve non-régression (`guimlab-proof-contract`).

---

## 5. ANNEXE — SIDELINE OPTIONNELLE : Modules d'extension Zig (HORS CONSTITUTION)

> ⚠️ **STATUT** : Cette annexe est un **plan d'extension exploratoire** qui ne modifie **PAS** la Règle d'Or §1 (C++20/CUDA bare-metal pour le **moteur** et le **runtime**).
> Les composants décrits ici sont strictement cantonnés aux **modules d'extension chargés dynamiquement** et aux **outils de support** (build, packaging, audit mémoire), jamais au hot-path.
> Toute proposition qui contreviendrait à cette séparation doit être rejetée par revue constitutionnelle.

### 5.1 Justification

Zig 0.14+ apporte trois bénéfices uniques **accessoires** (non-critiques pour le moteur) :

1. **Allocateur explicitement passé en argument** → détection déterministe des fuites mémoire au-delà d'ASan.
2. **Cross-compilation triviale single-binary** (`zig build -Dtarget=x86_64-linux-gnu -static`) → distribution edge/embedded sans glibc ni runtime CUDA host obligatoire.
3. **C-ABI natif** (`extern "c"`) → interop sans FFI avec le code C++20 existant via `add_custom_command(zig build-lib)`.

### 5.2 Périmètre EXCLUSIF (jamais dans le hot-path)

| Composant | Localisation | Statut constitutionnel |
|---|---|---|
| `guim_kernel.cu`, kernels CUDA résidents, Reflex L0 | `src/`, `include/` | 🚫 **INTouchable** — Constitution §1 |
| Allocator tracant pour build/dev | `tools/zig/tracking_allocator.zig` | ✅ Sideline |
| Packaging single-binary pour distribution | `tools/zig/bundle.zig` | ✅ Sideline |
| Module dynamique audio ingest (post-MVP) | `modules/audio_ingest/` (Zig .so/.dll) | ✅ Sideline |
| Module dynamique viewport studio (post-MVP) | `modules/viewport/` (Zig .so/.dll) | ✅ Sideline |
| Outil d'audit mémoire statique | `tools/zig/leak_audit.zig` | ✅ Sideline |

### 5.3 Roadmap en 4 phases (sideline, hors hot-path)

#### Phase 0 — Validation cohabitation (1-2 semaines) ⏳ EN ATTENTE
- **Livrable** : `tools/zig/hello_bridge.zig` + test CMake qui lie une `.lib` Zig statique à `guim_tests` sans régression.
- **Critère GO/NO-GO** : `ctest --output-on-failure` reste à 100% PASS avec le binaire Zig lié.
- **Risque** : 🟢 Faible — réversible à 100%.
- **Hors scope** : aucun fichier dans `src/` ou `include/` n'est modifié.

#### Phase 1 — Allocator tracant (2-4 semaines) ⏳ EN ATTENTE
- **Livrable** : `tools/zig/guim_allocator.zig` (TrackingAllocator avec peak/current/alloc/free counters atomiques).
- **Usage** : substitut à `malloc`/`free` dans les **binaires de support** (générateurs de figures, scripts de validation, packaging).
- **Critère succès** : `leak_audit.zig` détecte 100% des fuites introduites volontairement dans un test fixture.
- **Risque** : 🟢 Faible — usage limité aux outils.

#### Phase 2 — Bindings C-ABI depuis C++20 (1-2 mois) ⏳ EN ATTENTE
- **Livrable** : `include/guim_zig_abi.h` (header C pur) + `tools/zig/guim_bindings.zig` (export C-ABI).
- **Cible** : `src/studio/shm_client.cpp` et `src/studio/viewport_session.cpp` peuvent appeler des fonctions Zig via `extern "C"` **sans réécriture** — juste un wrapper léger.
- **Critère succès** : benchmark `< 5%` overhead vs C++ pur (FMA cache-line aligned).
- **Risque** : 🟡 Modéré — vérifier que les conventions d'appel x86_64 SysV / Windows x64 sont identiques.

#### Phase 3 — Modules dynamiques chargeables (3-6 mois) ⏳ EN ATTENTE
- **Livrable** : `modules/audio_ingest/` et `modules/viewport/` compilés en `.so`/`.dll` Zig, chargés via `dlopen`/`LoadLibrary` depuis `guim_node`.
- **Critère succès** : un module peut être remplacé sans recompilation du binaire principal ; allocateur tracant valide 0 leak.
- **Risque** : 🟡 Modéré — compatibilité binaire ABI à long terme (Zig 0.14 → 0.15 peut casser).

### 5.4 Garde-fous constitutionnels

- ❌ **INTERDIT** : réécrire `guim_kernel.cu` ou tout kernel CUDA résident en Zig.
- ❌ **INTERDIT** : remplacer un `std::vector`/`cudaMalloc` du hot-path par un allocateur Zig.
- ❌ **INTERDIT** : lier Zig dans `guim_kernel_tests` (le banc d'essai CUDA benchmarké).
- ❌ **INTERDIT** : utiliser Zig pour implémenter SR-MIT ou toute dynamique symplectique du cortex.
- ✅ **AUTORISÉ** : outils de build, packaging, audit mémoire, modules dynamiques périphériques.

### 5.5 Critères d'annulation (kill-switch)

Le plan Zig est **automatiquement abandonné** si l'un de ces seuils est franchi :
- Phase 0 : `guim_tests` perd > 0 tests PASS → STOP immédiat.
- Phase 1 : `leak_audit` produit des faux positifs > 5% → STOP, ré-évaluation.
- Phase 2 : overhead FFI > 10% vs C++ pur sur microbench `guim_bench` → STOP, retour à C++20.
- Phase 3 : un module dynamique cause une régression de stabilité mémoire (`0 bytes leak over 100,000 continuous frames` violé) → STOP, retour à C++20.

### 5.6 References

- Évaluation complète : voir `docs/12-zig-sideline-roadmap.md` (à créer en Phase 0).
- Zig 0.14 breaking changes : https://ziglang.org/documentation/0.14.0/
- Interop Zig↔C++20 : https://ziglang.org/learn/overview/#c-interop
- Audit scientifique adversarial : `tests/test_scientific_hypotheses.cpp` (modèle de rigueur à appliquer).

### 5.7 Statut actuel

| Phase | État | Démarrage estimé |
|---|---|---|
| Phase 0 | ⏳ **EN ATTENTE** (sideline backlog) | Après stabilisation du hot-path CUDA |
| Phase 1 | ⏳ **EN ATTENTE** | T2 — dépend Phase 0 GO |
| Phase 2 | ⏳ **EN ATTENTE** | T3 — dépend Phase 1 GO |
| Phase 3 | ⏳ **EN ATTENTE** | T4+ — dépend Phase 2 GO |

