# Spécification du projet : Plan or Learn? Dynamic Pricing of Airline Tickets

> Ce fichier est notre **contrat commun** : toutes les valeurs du code viennent d'ici.
> Statut : **VALIDÉE (Imane + Dia Eddine, 2026-10-09)**.
> Toute modification après validation se décide à deux, puis on met à jour la date ci-dessous.

Dernière mise à jour : 2026-10-09

---

## 1. Question de recherche

À partir de quelle erreur sur le modèle de demande devient-il préférable d'**apprendre sans modèle** (Q-learning) plutôt que de **planifier avec un modèle imparfait** (Dynamic Programming) ?

---

## 2. Le MDP

### 2.1 Convention de temps (important pour éviter les bugs)
- `t` = **nombre de jours restants** avant le départ.
- Un vol commence à `t = T` et se termine à `t = 0` (départ).
- À chaque pas : on fixe un prix pour la journée, les ventes ont lieu, puis `t ← t - 1`.

### 2.2 Éléments

| Élément | Valeur | Validé ? |
|---|---|---|
| Capacité `C` | 10 sièges (20 au départ, voir décision D1 dans `docs/decisions.docx`) | ✅ |
| Horizon `T` | 30 jours | ✅ |
| État `s` | `(c, t)` avec `c ∈ {0..C}` sièges restants, `t ∈ {0..T}` jours restants | ✅ |
| Actions | prix ∈ {50, 80, 110, 140, 170, 200} € (6 actions, indice 0 à 5) | ✅ |
| Récompense | `prix × ventes du jour` | ✅ |
| Fin d'épisode | `t = 0` (départ) **ou** `c = 0` (avion plein) | ✅ |
| `γ` | 1 (horizon fini). Variante E4 : 0.9 et 0.5 | ✅ |

**Taille :** (C+1) × (T+1) = 11 × 31 = 341 états, et 341 × 6 = 2 046 valeurs Q.

---

## 3. Le modèle de demande (le « vrai » environnement)

### 3.1 Deux segments de clients

| Segment | Arrivées par jour `λ(t)` | Prix max accepté (WTP) |
|---|---|---|
| Loisirs (L) | 0.8 si `t > 10`, 0.3 si `t ≤ 10` | exponentielle, moyenne **80 €** |
| Affaires (B) | 0.1 si `t > 10`, 1.0 si `t ≤ 10` | exponentielle, moyenne **180 €** |

- Un client achète si son **prix max accepté ≥ prix affiché**.
- Avec une WTP exponentielle de moyenne `m` : `P(achat | prix p) = exp(-p / m)`.

> ✏️ **À vérifier par nous** : nombre total moyen d'arrivées sur 30 jours (Loisirs ≈ ? , Affaires ≈ ?).
> Est-ce bien supérieur à 20 sièges à bas prix et inférieur à 20 à prix élevé ? Sinon, il n'y a pas de compromis intéressant.
>
> ✅ **Vérifié (2026-10-09)** : 19 arrivées loisirs + 12 affaires en moyenne. Même à 50 €, seulement 19,3 acheteurs : avec 20 sièges l'avion n'est presque jamais plein
> (une politique myope perd 0,1 % seulement). D'où **C = 10** : le myope perd alors 19 %, remplissage 77 %. Détails et argument : décision D1, `docs/decisions.docx`.

### 3.2 Nombre de ventes dans une journée
Propriété d'**amincissement de Poisson** : si les clients arrivent selon Poisson(λ) et que chacun achète avec la probabilité `q`, alors le nombre d'acheteurs suit **Poisson(λ · q)**.

Pour un jour `t` et un prix `p` :
- `μ(t, p) = λ_L(t) · q_L(p) + λ_B(t) · q_B(p)`
- `demande ~ Poisson(μ(t, p))`
- `ventes = min(demande, c)` (on ne vend pas plus que les sièges restants)

> ✅ `P(ventes = k) = P(demande = k)` pour `k < c`, et `P(ventes = c) = P(demande ≥ c)` : la dernière case
> « absorbe » toute la demande ≥ c (`DemandModel.sales_distribution`).

---

## 4. L'erreur de modèle (cœur du projet)

- La DP reçoit un `DemandModel` dont **un seul paramètre** est faussé.
- **Décision :** on multiplie **les deux moyennes de WTP** par `(1 + x)` (`DemandModel.with_error`), avec
  `x ∈ {-50 %, -30 %, -10 %, 0 %, +10 %, +30 %, +50 %}`.
- `x > 0` : la DP croit que les clients sont prêts à payer plus qu'en réalité.

> ✅ **Décidé (2026-10-09) : la WTP.** Une erreur sur la WTP coûte jusqu'à 24 % à la DP (75,8 % de l'optimum à x = −50 %),
> de façon asymétrique ; une erreur sur `λ` coûte au plus 5 %, la carte de E2 serait presque uniforme.
> Optionnel si le temps le permet : une ligne de E2 avec `λ` faussé, pour montrer la robustesse de la DP aux erreurs de volume.
> Détails : décision D6, `docs/decisions.docx`.

---

## 5. Méthodes comparées

1. **DP-vrai** : récurrence à rebours avec le vrai modèle → **optimum de référence**
2. **DP-faux** : même algorithme, modèle faussé (section 4)
3. **Q-learning** et **SARSA** tabulaires, ε-greedy décroissant
4. **Baselines** : meilleur prix fixe ; prix croissant à l'approche du départ

**Règle d'équité :** toutes les politiques sont évaluées par **la même fonction**, sur le **vrai** modèle.

---

## 6. Évaluation

| Élément | Valeur | Validé ? |
|---|---|---|
| Métrique principale | % du revenu optimal (revenu moyen / revenu DP-vrai) | ✅ |
| Métriques secondaires | taux de remplissage (load factor), probabilité d'avion plein, revenu moyen | ✅ |
| Seeds | 10 par configuration, dès le premier lancement (elles mesurent la variabilité de l'apprentissage) | ✅ |
| Évaluation d'une politique | **évaluation exacte** (`evaluate_exact`), Monte-Carlo 10 000 vols en vérification (`evaluate_mc`) | ✅ |

---

## 7. Expériences

| Id | Question | Responsable (sans répartition stricte) |
|---|---|---|
| E1 | Combien d'épisodes faut-il au Q-learning pour atteindre 95 % de l'optimum ? | Dia Eddine |
| E2 ⭐ | Carte du point de bascule : erreur de modèle × budget d'épisodes → qui gagne ? | ensemble |
| E3 | Cartes de chaleur des politiques (prix selon `c` et `t`) : DP vs Q-learning | Imane |
| E4 | Effet de `γ` (1, 0.9, 0.5) : l'agent devient-il myope ? | ensemble |

---

## 8. Tests de cohérence (à passer avant toute expérience)

1. Valeur DP-vrai `V_T(C)` ≈ revenu moyen simulé de sa politique (Monte-Carlo).
2. Avec `x = 0 %`, aucune méthode ne bat DP-vrai.
3. Q-learning avec beaucoup d'épisodes → proche de l'optimum.
4. Meilleur prix fixe < DP-vrai.

---

## 9. Décisions prises (réunion du 2026-10-09, issue #3)

- [x] Toutes les valeurs ci-dessus validées (C = 10 : décision D1)
- [x] Erreur de modèle : la WTP (section 4, décision D6)
- [x] Q-learning, choix de départ (fidèles au cours) : Q = 0 partout ; ε-greedy avec ε qui part de 1 puis décroît ;
      α constant ou décroissant ; budgets d'épisodes pour E2 sur une échelle logarithmique (≈ 100 → 100 000).
      Les valeurs finales de α et de la décroissance de ε sont fixées dans #14, puis reportées ici.
- [x] Interface de l'environnement : `step()` renvoie `(state, reward, done)` (sans `info`, contrairement au gym du cours)
- [x] Répartition : côté « Plan » (DP, baselines, DP-faux) pour Imane, côté « Learn » (Q-learning, SARSA, réglage) pour Dia Eddine ;
      on reste libres de prendre la prochaine issue débloquée.
- [ ] Valeurs finales des hyperparamètres du Q-learning (#14)
