# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:26:37 2022

@author: work
"""

from ast import main
from re import T
import pandas as pd
import numpy as np
import functools
import math
import numpy_financial as npf

class FunctionMemoizer:
    """
     Decorateur de Class pour implementation de Memoization :
         https://python-course.eu/advanced-python/memoization-decorators.php
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args, **kwargs):
        if args not in self.memo:
            self.memo[args] = self.fn(*args, **kwargs)
        return self.memo[args]

def MemoizeFunction(fn):
    """
    Decorateur de Fonction pour implementation de Memoization :
        https://python-course.eu/advanced-python/memoization-decorators.php

    Parameters
    ----------
    f : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    memo = {}
    def helper(*args):
        if args not in memo:
            memo[args] = fn(*args)
        return memo[args]
    return helper

def caching(func):
    """Keep a cache of previous function calls"""
    @functools.wraps(func)
    def wrapper_cache(*args, **kwargs):
        cache_key = args + tuple(kwargs.items())
        if cache_key not in wrapper_cache.cache:
            wrapper_cache.cache[cache_key] = func(*args, **kwargs)
        return wrapper_cache.cache[cache_key]
    wrapper_cache.cache = dict()
    return wrapper_cache

class cachingc:
    def __init__(self, fn):
        self.fn = fn
        self.cache = {}

    def __call__(self, *args, **kwargs):
        cache_key = args + tuple(kwargs.items())
        if cache_key not in self.cache:
            self.cache[cache_key] = self.fn(*args, **kwargs)
        return self.cache[cache_key]

class TblProd():
    def __init__(self, tblProduct) -> None:
        self.TblProduct = tblProduct

    def get_tx_prime_dc(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_prime_dc"].values[0]

    def get_tx_prime_inc(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_prime_inc"].values[0]

    def get_tx_prime_chomage(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_prime_chomage"].values[0]

    def get_tx_frais_admin(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_frais_admin"].values[0]

    def get_tx_frais_acq(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_frais_acq"].values[0]

    def get_tx_comm(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_comm"].values[0]

    def get_tx_profit_sharing_assureur(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_profit_sharing_assureur"].values[0]

    def get_tx_profit_sharing_partenaire(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_profit_sharing_partenaire"].values[0]

    def get_tx_production_financiere(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_production_financiere"].values[0]

    def get_tx_frais_gest_sin(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_frais_gest_sin"].values[0]

class LoiMaintienChomage():
    """
        Represente la loi de maintien en chômage entre t et t+1
    """
    def __init__(self, maintienCh) -> None:
        self.MaintienCh = maintienCh

    def  nombre_maintien_chomage(self, age_entree, anciennete_chomage):
        try:
            return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==math.floor(age_entree), str(anciennete_chomage)].values[0]
        except KeyError:
            return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==math.floor(age_entree), str(self.MaintienCh.shape[1]-2)].values[0]

    def prob_passage_ch_ch(self, age, anciennete_chomage):
        return self.nombre_maintien_chomage(age, anciennete_chomage+1)/self.nombre_maintien_chomage(age, anciennete_chomage)

class LoiMaintienIncapacite():
    """
        Represente la loi de maintien en incapacite entre t et t+1
    """
    def __init__(self, maintienIncap) -> None:
        self.MaintienIncap = maintienIncap

    def nombre_maintien_incap(self, age_entree, anciennete_incap):
        try:
            return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==math.floor(age_entree),str(anciennete_incap)].values[0]
        except KeyError:
            return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==math.floor(age_entree),str(self.MaintienIncap.shape[1]-2)].values[0]

    def prob_passage_inc_inc(self, age, anciennete_inc):
        return self.nombre_maintien_incap(age, anciennete_inc+1)/self.nombre_maintien_incap(age, anciennete_inc)

class LoiPasssageInvalidite():
    """
        Represente la loi de passage en invalidité entre t et t+1
    """
    def __init__(self, passageInval) -> None:
        self.PassageInval = passageInval

    def nombre_passage_inval(self, age_entree, anciennete_incap):
        try:
            return self.PassageInval.loc[self.PassageInval["Age_Anciennete"]==math.floor(age_entree),str(anciennete_incap)].values[0]
        except KeyError:
            return self.PassageInval.loc[self.PassageInval["Age_Anciennete"]==math.floor(age_entree),str(self.PassageInval.shape[1]-2)].values[0]

class LoiIncidence():
    """Probabilité des valides de passer en incap, ou inval, ou chomage, aussi appelée loi d'incidence
    """
    def __init__(self, incidence) -> None:
        self.Incidence = incidence

    def max_age_incidence(self):
        return max(self.Incidence["age_x"])

    def prob_entree_incap(self, age_actuel):
        if age_actuel<self.max_age_incidence():
            return self.Incidence.loc[self.Incidence["age_x"]==math.floor(age_actuel), "Incidence_en_incap"].values[0]
        else:
            return self.Incidence.loc[self.Incidence["age_x"]==self.max_age_incidence(), "Incidence_en_incap"].values[0]

    def prob_entree_chomage(self, age_actuel):
        if age_actuel < self.max_age_incidence():
            return self.Incidence.loc[self.Incidence["age_x"]==math.floor(age_actuel), "Incidence_en_chomage"].values[0]
        else:
            return self.Incidence.loc[self.Incidence["age_x"]==self.max_age_incidence(), "Incidence_en_chomage"].values[0]

    def prob_entree_inval(self, age_actuel):
        if age_actuel<self.max_age_incidence():
            return self.Incidence.loc[self.Incidence["age_x"]==math.floor(age_actuel), "Incidence_en_inval"].values[0]
        else:
            return self.Incidence.loc[self.Incidence["age_x"]==self.max_age_incidence(), "Incidence_en_inval"].values[0]

class LoiMortalite():
    """Loi de mortalité ou table de mortalité, probabilités pour un individu d'age x de decede avant l'age x+1
    """
    def __init__(self, mortaliteTH, mortaliteTF) -> None:
        self.MortaliteTH = mortaliteTH
        self.MortaliteTF = mortaliteTF

    def max_age_mortality_th(self):
        return max(self.MortaliteTH["age_x"])

    def max_age_mortality_tf(self):
        return max(self.MortaliteTF["age_x"])

    def prob_dc(self, sexe, age_actuel):
        if sexe=='F':
            if age_actuel>=self.max_age_mortality_tf():
                return self.MortaliteTF.loc[self.MortaliteTF["age_x"]==self.max_age_mortality_tf(), "Qx"].values[0]
            else :
                return self.MortaliteTF.loc[self.MortaliteTF["age_x"]==math.floor(age_actuel), "Qx"].values[0]
        else:
            if age_actuel>=self.max_age_mortality_th():
                return self.MortaliteTH.loc[self.MortaliteTH["age_x"]==self.max_age_mortality_th(), "Qx"].values[0]
            else :
                return self.MortaliteTH.loc[self.MortaliteTH["age_x"]==math.floor(age_actuel), "Qx"].values[0]

class LoiRachat():
    def __init__(self, lapse) -> None:
        self.Lapse = lapse

    def prob_rachat(self, produit, anciennete_contrat_mois):
        try:
            return self.Lapse.loc[self.Lapse["produit"]==produit, str(anciennete_contrat_mois)].values[0]
        except KeyError:
            return self.Lapse.loc[self.Lapse["produit"]==produit, str(self.Lapse.shape[1]-2)].values[0]

class ADEFlux():
    """_summary_
        Class projection to generate projection objects
        Modele de markov :
            INC -> INC
            INC -> INV
            INC -> VALIDE
            INC -> DC

            VALLIDE -> CH
            VALIDE -> INV
            VALIDE -> INC
            VALIDE -> DC
            VALIDE -> VALIDE

            CH -> CH
            CH -> VALIDE
            CH -> DC

            INV -> INV
            INC -> DC
    """

    def __init__(self, modelPointRow) -> None:
        self.ModelPointRow = modelPointRow

    IncidenceData = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//INCIDENCE.csv', sep=";")
    LapseData = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//LAPSE.csv', sep=";")
    MaintienChData = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MAINTIEN_CH.csv', sep=";")
    MaintienIncapData = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MAINTIEN_INCAP.csv', sep=";")
    MortaliteTHData = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MORTALITE_TF0002.csv', sep=";")
    MortaliteTFData = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MORTALITE_TH0002.csv', sep=";")
    PassageInvalData = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//PASSAGE_INVAL.csv', sep=";")
    referentielProduit = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//TBL_PROD.csv', sep=";")

    Incidence = LoiIncidence(IncidenceData)
    Lapse = LoiRachat(LapseData)
    MaintienCh = LoiMaintienChomage(MaintienChData)
    MaintienIncap = LoiMaintienIncapacite(MaintienIncapData)
    Mortalite = LoiMortalite(MortaliteTHData, MortaliteTFData)
    PassageInval = LoiPasssageInvalidite(PassageInvalData)
    ReferentielProduit = TblProd(referentielProduit)

    def mp_id(self):
        return self.ModelPointRow.mp_id

    def sexe(self):
        return self.ModelPointRow.sexe

    def etat(self):
        return self.ModelPointRow.etat

    @functools.lru_cache
    def age_actuel(self, t):
        return self.ModelPointRow.age_actuel + t/12

    def age_souscription_annee(self):
        return self.ModelPointRow.age_souscription_annee

    def annee_souscription(self):
        return self.ModelPointRow.annee_souscription

    @functools.lru_cache
    def anciennete_contrat_annee(self,t):
        return self.ModelPointRow.anciennete_contrat_annee + t/12

    @functools.lru_cache
    def anciennete_contrat_mois(self,t):
        return self.ModelPointRow.anciennete_contrat_mois + t

    def duree_pret(self):
        return self.ModelPointRow.duree_pret

    def age_fin(self):
        return self.ModelPointRow.age_fin

    def ci(self):
        """
            retourne le Capital Initial du prêt
        Returns:
            float: Capital initial du prêt
        """
        return self.ModelPointRow.ci

    def crd(self, t):
        """
            retourne le Montant Restant dû du crédit juste après la mensualité versé à l'instant t
        Returns:
            float: montant du capital restant dû
        """
        crd = self.loan_balance_at_n(self.ci(), self.taux_nominal(), 12, self.duree_pret()/12, (self.duree_pret()-self.duree_restante(t)))
        return crd

    @functools.lru_cache
    def duree_restante(self,t):
        return self.ModelPointRow.duree_restante - t

    def taux_nominal(self):
        return self.ModelPointRow.taux_nominal

    def taux_mensuel(self):
        return self.ModelPointRow.taux_mensuel

    def mensualite(self):
        return self.ModelPointRow.mensualite

    def prime_dc(self):
        return self.ModelPointRow.prime_dc

    def prime_inc_inv(self):
        return self.ModelPointRow.prime_inc_inv

    def prime_ch(self):
        return self.ModelPointRow.prime_ch

    def produit(self):
        return self.ModelPointRow.produit

    def distribution_channel(self):
        return self.ModelPointRow.distribution_channel

    def nb_contrats(self):
        return self.ModelPointRow.nb_contrats
    
    def age_entree_sinistre(self, t):
        return self.age_actuel(t) - self.duree_sinistre(t)

    @functools.lru_cache
    def duree_sinistre(self,t):
        return self.ModelPointRow.duree_sinistre + t

    @functools.lru_cache
    def prob_passage_inc_inv(self, age_entree, duration_incap):
        return self.PassageInval.nombre_passage_inval(age_entree, duration_incap)/self.MaintienIncap.nombre_maintien_incap(age_entree, duration_incap)

    @functools.lru_cache
    def fibonacci(self, num):
        if num < 2:
            return num
        return self.fibonacci(num - 1) + self.fibonacci(num - 2)
    
    def projection_initiale_state_at_t(self, init_state_at_t0, actual_state_at_t_n_1, mat_transitions):
        res = np.zeros((2, mat_transitions.shape[1]))
        tra = np.dot(actual_state_at_t_n_1, mat_transitions)
        res[0,] = (init_state_at_t0>0)*tra
        res[1,] = (init_state_at_t0==0)*tra
        return res
    
    def constitution_matrix_transitions(self, age_entree_etat, duration_etat_at_t, t):
        """Calcul de la matrice des probabilité de transitions pour les 6 états : 
        [VALIDE, DC, CHOMAGE, INCAPACITE, INVALIDITE, LAPSE]    x   [VALIDE, DC, CHOMAGE, INCAPACITE, INVALIDITE, LAPSE]
        les états absorbants ont une probabilité totale de changement d'état = 0 (au lieu de 1) : somme(i,j): j -> 1 à n, i constante = 0.
        Ce qui permet de calculer uniquement à l'instant t l'effectif des autres états qui ont transité dans un état absorbant et non le cumul
        depuis le début de la projection
        
        Args:
            age_entree_etat (int): age d'entrée dans l'état en années
            duration_etat_at_t (int): duration dans l'état à l'instant t en mois
            t (int): instant t de projection

        Returns:
            Matrice: Matrice carrée des proba de transitions avec l'élément [i,j] est la proba de passer de l'état i à l'état j
        """
        # initialisation de la matrice des transtions à zeros
        sex = self.sexe()
        age_actuel = self.age_actuel(t)
        produit = self.produit()
        anc_contrat_mois = self.anciennete_contrat_mois(t)
        mat = np.zeros((6, 6))
        # From valide to ...
        mat[0,1] = ADEFlux.Mortalite.prob_dc(sex, age_actuel)/12
        mat[0,2] = ADEFlux.Incidence.prob_entree_chomage(age_actuel)/12
        mat[0,3] = ADEFlux.Incidence.prob_entree_incap(age_actuel)/12
        mat[0,4] = ADEFlux.Incidence.prob_entree_inval(age_actuel)/12
        mat[0,5] = ADEFlux.Lapse.prob_rachat(produit , anc_contrat_mois)
        mat[0,0] = 1 - np.sum(mat[0,:])
        # From DC to ...
        mat[1,1] = 0
        # From Chomage to ...
        mat[2,1] = ADEFlux.Mortalite.prob_dc(sex, age_actuel)/12
        mat[2,2] = ADEFlux.MaintienCh.prob_passage_ch_ch(age_entree_etat, duration_etat_at_t)
        mat[2,0] = 1 - np.sum(mat[2,:])
        # From incap to ...
        mat[3,1] = ADEFlux.Mortalite.prob_dc(sex, age_actuel)/12
        mat[3,3] = ADEFlux.MaintienIncap.prob_passage_inc_inc(age_entree_etat, duration_etat_at_t)
        mat[3,4] = self.prob_passage_inc_inv(age_entree_etat, duration_etat_at_t)
        mat[3,0] = 1 - np.sum(mat[3,:])
        # From invalidite to ...
        mat[4,4] = 0
        # From Lapse to ...
        mat[5,5] = 0
        return mat

    @functools.lru_cache
    def get_next_state(self, t):
        """Vieillissement des effectifs

        Args:
            t (int): temps t de projection

        Returns:
            matrice: repartitions des effectis par anciennetée
        """
        
        if t == 0:
            return self.vecteur_des_effectifs_at_t(0)
        else:
            mat_transitions = self.constitution_matrix_transitions(self.age_entree_sinistre(t), self.duree_sinistre(t), t)
            init_state_projection = self.projection_initial_state_at_t(self.vecteur_des_effectifs_at_t(0), self.get_next_state(t-1)[0,:], mat_transitions)
            for row_i in range(1, self.get_next_state(t-1).shape[0]):
                mat_transitions = self.constitution_matrix_transitions(self.age_entree_sinistre(t), row_i, t) # durée sinistre = row_i pour modéliser les nouvelles transtions
                project_at_t = np.dot(self.get_next_state(t-1)[row_i,:], mat_transitions)
                init_state_projection = np.vstack([init_state_projection, project_at_t.reshape(1, mat_transitions.shape[0])])
            init_state_projection[0,0] = np.sum(init_state_projection[:,0])
            init_state_projection[1:,0] = 0
        return init_state_projection
    
    @functools.lru_cache
    def vecteur_des_effectifs_at_t(self, t):
        """Vecteur de effectifs :
            6 états suivants : VALIDE, DC, CHOMAGE, INCAPACITE, INVALIDITE, LAPSE

        Returns:
            vecteur : Effectifs par état sour la forme [Nombre de VALIDE, Nombre de DC, Nombre de CHOMAGE, Nombre de INCAPACITE, Nombre de INVALIDITE, Nombre de LAPSE]
        """
        if t == 0:
            if self.etat()=='v':
                return self.nb_contrats(0) * np.array([[1, 0, 0, 0, 0, 0]])
            if self.etat()=='ch':
                return self.nb_contrats(0) * np.array([[0, 0, 1, 1, 0, 0]])
            if self.etat()=='inc':
                return self.nb_contrats(0) * np.array([[0, 0, 0, 1, 0, 0]])
        else:
            return  np.sum(self.get_next_state(t), axis=0)

    @functools.lru_cache
    def nombre_de_v(self, t):
        if t ==0:
            return self.nb_contrats() if self.etat() == 'v' else 0
        else:
            proba_v_v = 1 - ADEFlux.Lapse.prob_rachat(self.produit(), self.anciennete_contrat_mois(t)) - ADEFlux.Mortalite.prob_dc(self.sexe(), self.age_actuel(t)) - ADEFlux.Incidence.prob_entree_chomage(self.age_actuel(t)) - ADEFlux.Incidence.prob_entree_incap(self.age_actuel(t))
            proba_inc_v = 1 - ADEFlux.Mortalite.prob_dc(self.sexe(), self.age_actuel(t)) - ADEFlux.MaintienIncap.prob_passage_inc_inc(self.age_actuel(t), self.duree_sinistre(t)) - self.prob_passage_inc_inv(t)
            proba_ch_v = 1 - ADEFlux.Mortalite.prob_dc(self.sexe(), self.age_actuel(t)) - ADEFlux.MaintienCh.nombre_maintien_chomage(self.age_actuel(t), self.duree_sinistre(t))
            return 10 #self.nombre_de_v(t-1) * proba_v_v # + self.nombre_de_inc(t-1) * proba_inc_v + self.nombre_de_ch(t-1) * proba_ch_v

    @functools.lru_cache
    def nombre_de_ch(self, t):
        if t ==0:
            return self.nb_contrats() if self.etat() == 'ch' else 0
        else:
            return self.nombre_de_v(t-1) * ADEFlux.Incidence.prob_entree_chomage(self.age_actuel(t)) + ADEFlux.MaintienCh.prob_passage_ch_ch(self.age_actuel(t), self.duree_sinistre(t)) * self.nombre_de_ch(t-1)

    @functools.lru_cache
    def nombre_de_inv(self, t):
        if t == 0:
            return self.nb_contrats() if self.etat() == 'inv' else 0
        else:
            return self.nombre_de_inv(t-1) + self.prob_passage_inc_inv(t) * self.nombre_de_inc(t-1) + ADEFlux.Incidence.prob_entree_inval(self.age_actuel(t)) * self.nombre_de_v(t-1)

    @functools.lru_cache
    def nombre_de_inc(self, t):
        if t == 0:
            return self.nb_contrats() if self.etat() == 'inc' else 0
        else:
            return ADEFlux.MaintienIncap.prob_passage_inc_inc(self.age_actuel(t), self.duree_sinistre(t)) * self.nombre_de_inc(t-1) + (ADEFlux.Incidence.prob_entree_incap(self.age_actuel(t)) / 12) * self.nombre_de_v(t-1)

    @functools.lru_cache
    def nombre_de_dc(self, t):
        if t == 0:
            return 0
        else:
            return self.nombre_de_dc(t-1) + ADEFlux.Mortalite.prob_dc(self.sexe(), self.age_actuel(t)) / 12 * (self.nombre_de_inc(t-1) + self.nombre_de_inv(t-1) + self.nombre_de_ch( t-1) + self.nombre_de_v(t-1))

    @functools.lru_cache
    def nombre_de_lps(self, t):
        if t == 0:
            return 0
        return self.nombre_de_lps(t-1) + ADEFlux.Lapse.prob_rachat(self.produit(), self.anciennete_contrat_mois(t)) * self.nombre_de_v(t-1)

    def pmxcho(self, age_entre, dure_ecoulee, D1, D2, taux_actu):
        som1 = 0
        som2 = 0
        l = ADEFlux.MaintienCh.nombre_maintien_chomage(age_entre, dure_ecoulee)
        for i in range(D1,D2):
            som1 = som1 + ((1 + taux_actu)^(-(i / 12))) * ADEFlux.MaintienCh.nombre_maintien_chomage(age_entre, dure_ecoulee + i)
            som2 = som2 + ((1 + taux_actu)^(-((i + 1) / 12))) * ADEFlux.MaintienCh.nombre_maintien_chomage(age_entre, dure_ecoulee + i + 1)
        return((som1 + som2) / (2 * l))

    def pmxinc(agentree, durecoulee, D1, D2, taux_actu):
        som1 = som2 = 0
        l = ADEFlux.MaintienIncap.nombre_maintien_incap(agentree, durecoulee)
        for i in range(D1,D2):
            som1 = som1 + ((1 + taux_actu)^(-(i / 12))) * ADEFlux.MaintienIncap.nombre_maintien_incap(agentree, durecoulee + i)
            som2 = som2 + ((1 + taux_actu)^(-((i + 1) / 12))) * ADEFlux.MaintienIncap.nombre_maintien_incap(agentree, durecoulee + i + 1)
        return ((som1 + som2) / (2 * l))

    def pmxpot2(agentree, durecoulee, D1, D2, taux, crd):
        som1 = som2 = 0
        l = ADEFlux.MaintienIncap.nombre_maintien_incap(agentree, durecoulee)
        prov = crd
        for i in range(D1,D2):
            som1 = som1 + ((1 + taux) ^ -(i / 12)) * ADEFlux.PassageInval.nombre_passage_inval(agentree, durecoulee + i) * prov
            som2 = som2 + ((1 + taux) ^ -((i + 1) / 12)) * ADEFlux.PassageInval.nombre_passage_inval(agentree, durecoulee + i + 1) * prov
        return ((som1 + som2) / (2 * l))

    def amortisation_schedule(self, amount, annualinterestrate, paymentsperyear, years):
        """
            Tableau d'amortissement du prêt
        Args:
            amount (float): Montant du prêt
            annualinterestrate (float): taux d'interet annuel
            paymentsperyear (float): Nombre de paiment dans l'année
            years (int): nombre d'années du prêt

        Returns:
            Pandas DataFrame: Tableau d'amortissement
        """
        df = pd.DataFrame({'PrincipalPaid' :[npf.ppmt(annualinterestrate/paymentsperyear, i+1, paymentsperyear*years, amount) for i in range(paymentsperyear*years)],
                           'InterestPaid' :[npf.ipmt(annualinterestrate/paymentsperyear, i+1, paymentsperyear*years, amount) for i in range(paymentsperyear*years)]})
        df['Instalment'] = df.PrincipalPaid + df.InterestPaid
        df['CumulativePrincipal'] = np.cumsum(df.PrincipalPaid)
        df['Principal'] = amount
        df['Balance'] = df['Principal'] + df['CumulativePrincipal']
        df['Mois'] = np.arange(1, df.shape[0]+1, 1)
        return (df)
    
    def loan_balance_at_n(self, amount, annualinterestrate, paymentsperyear, years, n):
        principalpaid = [npf.ppmt(annualinterestrate/paymentsperyear, i+1, paymentsperyear*years, amount) for i in range(n)]
        return amount+np.sum(principalpaid)

    def results(self):
        """Projection du model point à la fin du contrat et ou des garanties

        Returns:
            DataFrame: variables du mp projetées dans le temps
        """
        ultim = self.duree_restante(0)
        df = pd.DataFrame({'time':[t for t in range(ultim)],
                           'mp_id':[self.mp_id() for t in range(ultim)],
                           'sexe':[self.sexe() for t in range(ultim)],
                           'etat':[self.etat() for t in range(ultim)],
                           'age_actuel':[self.age_actuel(t)  for t in range(ultim)],
                           'age_souscription_annee':[self.age_souscription_annee() for t in range(ultim)],
                           'annee_souscription':[self.annee_souscription() for t in range(ultim)],
                           'anciennete_contrat_annee':[self.anciennete_contrat_annee(t) for t in range(ultim)],
                           'anciennete_contrat_mois':[self.anciennete_contrat_mois(t) for t in range(ultim)],
                           'duree_pret':[self.duree_pret() for t in range(ultim)],
                           'age_fin':[self.age_fin() for t in range(ultim)],
                           'ci':[self.ci() for t in range(ultim)],
                           'crd':[self.crd(t)  for t in range(ultim)],
                           'duree_restante':[self.duree_restante(t) for t in range(ultim)],
                           'taux_nominal':[self.taux_nominal() for t in range(ultim)],
                           'taux_mensuel':[self.taux_mensuel() for t in range(ultim)],
                           'mensualite':[self.mensualite() for t in range(ultim)],
                           'prime_dc':[self.prime_dc() for t in range(ultim)],
                           'prime_inc_inv':[self.prime_inc_inv() for t in range(ultim)],
                           'prime_ch':[self.prime_ch() for t in range(ultim)],
                           'produit':[self.produit() for t in range(ultim)],
                           'distribution_channel':[self.distribution_channel() for t in range(ultim)],
                           'nb_contrats':[self.nb_contrats() for t in range(ultim)],
                           'duree_sinistre':[self.duree_sinistre(t) for t in range(ultim)]})
        return df

if __name__=="__main__":
    #data_files_path ='C:/Users/work/OneDrive/modele_emprunteur/CSV'
    ModelPoint = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MODEL_POINT.csv', sep=";")
    projection = ADEFlux(ModelPoint.loc[0,:])
    projection.get_next_state(1)