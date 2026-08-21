# Ex08 — Аудит: базовий метод → модифікація (11 методів)

> Перевірено за вихідним кодом кожного метода

## Наші 7 методів

| Метод | Тип | Базовий метод (prior art) | Що саме модифіковано |
|-------|-----|--------------------------|---------------------|
| **TESA-26** | Модифікація | **Taylor FO** (Molchanov et al., 2019) — базовий скоринг Sᵢⱼ=‖Wᵢⱼ · gᵢⱼ‖ | +ISR (перерахунок saliency з маскою), +NFP (штраф за мертві нейрони), +CMA-ES для пошарового розподілу бюджету |
| **SET-v2** | Модифікація | **SET** (Mocanu et al., 2018) — Sparse Evolutionary Training | Ініціалізація масок через TESA-26 замість випадкової; імпортує `_compute_saliency`, `_masks_from_k` з `tesa26.py` |
| **FES-NSDE** | Модифікація | **DE** (Storn & Price, 1997) + **Fisher-Taylor 2nd order** (Molchanov, 2019; Theis et al., 2018) | Скоринг: ‖W · E[g]‖ + 0.5 W² E[g²] (Fisher-Taylor 2-го порядку). Еволюція: DE/rand/1/bin у null-space з Lagrangian repair |
| **ACDE** | Модифікація | **DE** + **Aitchison geometry** (Aitchison, 1986) + **Fisher-Taylor 2nd order** | Мутація в log-ratio просторі симплекса: p₁ · (p₂/p₃)^F. Скоринг = той самий Fisher-Taylor 2-го порядку що й FES-NSDE |
| **E-HTA** | Модифікація | **Taylor FO** (Molchanov, 2019) + **OBD** (LeCun et al., 1990) | Скоринг: ‖g · w‖ + λ w², де λ еволюціонується CMA-ES per-layer. Базелін: GraSP (Wang, ICLR 2020) використовує фіксований Гессіан |
| **VPAM** | Комбінація | **RIA** (Zhang, ICLR 2024) + **WANDA** (Sun, 2024) | rᵢⱼ з RIA + ‖X‖₂ з WANDA + 2 нові: exp(-Var) та адаптивний per-row бюджет |
| **EvoSynFlow** | Модифікація | **SynFlow** (Tanaka et al., 2020) — data-free потік синаптичних сигналів | Ітеративний SynFlow (Ex07) + CMA-ES оптимізація пошарових коефіцієнтів |

## Базелін та SOTA (4 методи — реалізовані без змін)

| Метод | Джерело |
|-------|---------|
| Magnitude | Han et al. (2015) |
| WANDA | Sun et al. (2024) |
| SparseGPT | Frantar & Alistarh (2023) |
| RIA | Zhang et al. (ICLR 2024) |

## Загальна схема модифікацій

```
Prior art (скоринг)        Наша модифікація (еволюція + нові компоненти)
─────────────────────      ──────────────────────────────────────────────
Taylor FO (Molchanov)  ──→ TESA-26 (+ISR, +NFP, +CMA-ES)
                       ──→ E-PQM (у архіві; +phase quantization)

Taylor FO + OBD        ──→ E-HTA (+CMA-ES λ per-layer)

Fisher-Taylor 2nd ord  ──→ FES-NSDE (+DE/null-space repair)
                       ──→ ACDE (+Aitchison simplicial DE)

RIA + WANDA            ──→ VPAM (+exp(-Var), +adaptive budget)

SynFlow (Tanaka)       ──→ EvoSynFlow (+iterative, +CMA-ES)

SET (Mocanu)           ──→ SET-v2 (+TESA-26 initialization)
```

## Висновок

**Жоден метод НЕ є повністю оригінальним з нуля.** Всі 7 наших методів — це модифікації існуючих підходів з додаванням еволюційних операторів та/або нових компонентів (ISR, NFP, Aitchison geometry, exp(-Var)).

**Ключова новизна** полягає не у скорингу (він запозичений), а у:
1. **Еволюційному пошуку пошарового розподілу бюджету** (CMA-ES / DE)
2. **Нових компонентах фітнес-функції** (NFP, ISR, compositional mutation)
3. **Новій комбінаторній формулі** (VPAM = RIA ⊕ WANDA ⊕ exp(-Var))
