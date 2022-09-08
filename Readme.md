# Documentation : Modele assurance de l'emprunteur


## Introduction

Modele de projection des cash flow d'un effectif de models point d'un portefeuille assurance emprunteur. Le modele est developpé en programmation orientée objet avec des méthodes récursives (algorithmes récursifs) donc les différentes exécutions sont mises en cache avec le decorateur @functools.lru_cache du package functools de python. L'objectif de la mise en cache, des résultats d'éxécutions des méthodes est d'accélérer les calculs (la ***memoization***). 

## Projection des effectifs

La projection des effectifs est faite avec une chaine de markov.

### Les états

Modèle à 6 états : VALIDE, DC, CHOMAGE, INCAPACITE, INVALIDITE, LAPSE avec DC, INVALIDITE, LAPSE des états absorbants.
Codes dans les models points : v : valide, ch : chomage, inc : incapacite.

### Matrice des transitions

Elle représente les probabilités de passage entre états :

  |                | VALIDE | DC  | CHOMAGE | INCAPACITE | INVALIDITE | LAPSE |
  | -------------- | ------ | --- | ------- | ---------- | ---------- | ----- |
  | **VALIDE**     |        |     |         |            |            |       |
  | **DC**         |        |     |         |            |            |       |
  | **CHOMAGE**    |        |     |         |            |            |       |
  | **INCAPACITE** |        |     |         |            |            |       |
  | **INVALIDITE** |        |     |         |            |            |       |
  | **LAPSE**      |        |     |         |            |            |       |

  ### Représention graphique en chaine de markow

![image info](./CSV/Transitions-1.jpg)

  ### Mecanisme de projection des effectifs

  Soit le Vecteur des effectifs à $t=0$; $V(0)$.
  exemple : $V(0) = 1 valide$

  | VALIDE | DC  | CHOMAGE | INCAPACITE | INVALIDITE | LAPSE |
  | ------ | --- | ------- | ---------- | ---------- | ----- |
  | 1      | 0   | 0       | 0          | 0          | 0     |

  Soit la matrice des transitions entre $t-1$ et $t$ : $M(t)$
  La Matrice de transition $M(t)$ est calculée à partir des différentes lois de transitions et d'incidences. ces lois, dans ce modèle sont fonction de l'âge, l'âge d'entré dans un état ainsi que la duration effectuée dans cet état.
  On a Le vecteur des effectifs à $t$ : $  V(t) = V(t-1) * M(t) $

##### Schema de projection des effectifs :

  ![image info](./CSV/projection_effectifs.JPG)
