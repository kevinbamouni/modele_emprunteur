# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:26:37 2022

@author: work
"""

import pandas as pd
import numpy as np
import functools

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
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_prime_dc"]

    def get_tx_prime_inc(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_prime_inc"]

    def get_tx_prime_chomage(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_prime_chomage"]

    def get_tx_frais_admin(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_frais_admin"]

    def get_tx_frais_acq(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_frais_acq"]

    def get_tx_comm(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_comm"]

    def get_tx_profit_sharing_assureur(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_profit_sharing_assureur"]

    def get_tx_profit_sharing_partenaire(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_profit_sharing_partenaire"]

    def get_tx_production_financiere(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_production_financiere"]

    def get_tx_frais_gest_sin(self, productCode):
        return self.TblProduct[self.TblProduct["produits"]==productCode,"tx_frais_gest_sin"]

class LoiMaintienChomage():
    """
        Represente la loi de maintien en chômage
    """
    def __init__(self, maintienCh) -> None:
        self.MaintienCh = maintienCh

    def  nombre_maintien_chomage(self, age_entre, anciennete_chomage):
        return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==age_entre, str(anciennete_chomage)]


    def prob_passage_ch_ch(self, age, anciennete_chomage):
        return self.nombre_maintien_chomage(age, anciennete_chomage+1)/self.nombre_maintien_chomage(age, anciennete_chomage)

class LoiMaintienIncapacite():
    """
        Represente la loi de maintien en chômage
    """
    def __init__(self, maintienIncap) -> None:
        self.MaintienIncap = maintienIncap

    def nombre_maintien_incap(self, age_entre, anciennete_incap):
        return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==age_entre,str(anciennete_incap)]

    def prob_passage_inc_inc(self, age, anciennete_inc):
        return self.nombre_maintien_incap(age, anciennete_inc+1)/self.nombre_maintien_incap(age, anciennete_inc)

class LoiPasssageInvalidite():
    """
        Represente la loi de passage en invalidité
    """
    def __init__(self, passageInval) -> None:
        self.PassageInval = passageInval

    def nombre_passage_inval(self, age_entree, anciennete_incap):
        return self.PassageInval.loc[self.PassageInval["Age_Anciennete"]==age_entree,str(anciennete_incap)]

class LoiIncidence():
    def __init__(self, incidence) -> None:
        self.Incidence = incidence

    def max_age_incidence(self):
        return max(self.Incidence["age_x"])

    def prob_entree_incap(self, age_actuel):
        if age_actuel<self.max_age_incidence():
            return self.Incidence.loc[ADEFlux.Incidence["age_x"]==age_actuel, "Incidence_en_incap"]
        else:
            return self.Incidence.loc[ADEFlux.Incidence["age_x"]==self.max_age_incidence(), "Incidence_en_incap"]

    def prob_entree_chomage(self, age_actuel):
        if age_actuel < self.max_age_incidence():
            return self.Incidence.loc[ADEFlux.Incidence["age_x"]==age_actuel, "Incidence_en_chomage"]
        else:
            return self.Incidence.loc[ADEFlux.Incidence["age_x"]==self.max_age_incidence(), "Incidence_en_chomage"]

    def prob_entree_inval(self, age_actuel):
        if age_actuel<self.max_age_incidence():
            return self.Incidence.loc[ADEFlux.Incidence["age_x"]==age_actuel, "Incidence_en_inval"]
        else:
            return self.Incidence.loc[ADEFlux.Incidence["age_x"]==self.max_age_incidence(), "Incidence_en_inval"]

class LoiMortalite():
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
                return self.MortaliteTF.loc[self.MortaliteTF["age_x"]==self.max_age_mortality_tf(), "Qx"]
            else :
                return self.MortaliteTF.loc[self.MortaliteTF["age_x"]==age_actuel, "Qx"]
        else:
            if age_actuel>=self.max_age_mortality_th():
                return self.MortaliteTH.loc[self.MortaliteTH["age_x"]==self.max_age_mortality_th(), "Qx"]
            else :
                return self.MortaliteTH.loc[self.MortaliteTH["age_x"]==age_actuel, "Qx"]

class LoiRachat():
    def __init__(self, lapse) -> None:
        self.Lapse = lapse

    def prob_rachat(self, produit, anciennete_contrat_mois):
        return self.Lapse.loc[self.Lapse["produit"]==produit, str(anciennete_contrat_mois)]

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

    def age_actuel(self, t):
        return self.ModelPointRow.age_actuel + t/12

    def age_souscription_annee(self):
        return self.ModelPointRow.age_souscription_annee

    def annee_souscription(self):
        return self.ModelPointRow.annee_souscription

    def anciennete_contrat_annee(self,t):
        return self.ModelPointRow.anciennete_contrat_annee + t/12

    def anciennete_contrat_mois(self,t):
        return self.ModelPointRow.anciennete_contrat_mois + t

    def duree_pret(self):
        return self.ModelPointRow.duree_pret

    def age_fin(self):
        return self.ModelPointRow.age_fin

    def ci(self):
        return self.ModelPointRow.ci

    def crd(self):
        return self.ModelPointRow.crd

    @cachingc
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

    def duree_sinistre(self,t):
        return self.ModelPointRow.duree_sinistre + t

    def prob_passage_inc_inv(self, age, anciennete_inc):
        return self.PassageInval.nombre_passage_inval(age, anciennete_inc)/self.MaintienIncap.nombre_maintien_incap(age, anciennete_inc)

    @functools.lru_cache
    def fibonacci(self, num):
        if num < 2:
            return num
        return self.fibonacci(num - 1) + self.fibonacci(num - 2)

    @functools.lru_cache
    def nombre_de_v(self, t):
        if t ==0:
            return lambda: self.nb_contrats if self.etat() == 'v' else 0
        else:
            return self.nombre_de_v(t-1) - self.nombre_de_ch(t-1) - self.nombre_de_inv(t-1) - self.nombre_de_inc(t-1) - self.nombre_de_dc(t-1) - self.nombre_de_lps(t-1)

    @functools.lru_cache
    def nombre_de_ch(self, t):
        if t ==0:
            return lambda: self.nb_contrats if self.etat() == 'ch' else 0
        else:
            return self.nombre_de_v(t-1) * self.Incidence.prob_entree_chomage(self.age_actuel(t)) + self.MaintienCh.prob_passage_ch_ch(self.age_actuel(t), self.duree_sinistre(t)) * self.nombre_de_ch(t-1)

    @functools.lru_cache
    def nombre_de_inv(self, t):
        if t == 0:
            return lambda: self.nb_contrats if self.etat() == 'inv' else 0
        else:
            return self.nombre_de_inv(t-1) + self.prob_passage_inc_inv(self.age_actuel(t), self.duree_sinistre(t)) * self.nombre_de_inc(t-1) + self.Incidence.prob_entree_inval(self.age_actuel(t)) * self.nombre_de_v(t-1)

    @functools.lru_cache
    def nombre_de_inc(self, t):
        if t == 0:
            return lambda: self.nb_contrats if self.etat() == 'inc' else 0
        else:
            return self.MaintienIncap.prob_passage_inc_inc(self.age_actuel(t), self.duree_sinistre(t)) * self.effectifs_par_etat( t-1)["inc"] + self.Incidence.prob_entree_incap(self.age_actuel(t)) / 12 * self.effectifs_par_etat( t-1)["v"]

    @functools.lru_cache
    def nombre_de_dc(self, t):
        if t == 0:
            return 0
        else:
            return self.nombre_de_dc(t-1) + self.Mortalite.prob_dc(self.sexe(), self.age_actuel(t)) / 12 * (self.nombre_de_inc(t-1) + self.nombre_de_inv(t-1) + self.nombre_de_ch( t-1) + self.nombre_de_v(t-1))

    @functools.lru_cache
    def nombre_de_lps(self, t):
        if t == 0:
            return 0
        else:
            return self.nombre_de_lps(t-1) + self.Lapse.prob_rachat(self.produit(), self.anciennete_contrat_mois(t)) * self.nombre_de_v(t-1)

    @functools.lru_cache
    def effectifs_par_etat(self, t):
        if t==0:
            df = pd.DataFrame({})
            df["v"] = [lambda: self.nb_contrats if self.etat() == 'v' else 0]
            df["ch"] = [lambda: self.nb_contrats if self.etat() == 'ch' else 0]
            df["inv"] = [lambda: self.nb_contrats if self.etat() == 'inv' else 0]
            df["inc"] = [lambda: self.nb_contrats if self.etat() == 'inc' else 0]
            df["dc"] = [0]
            df["lps"] = [0]
            return df
        else:
            df["ch"] = self.effectifs_par_etat(t-1)["v"] * self.Incidence.prob_entree_chomage(self.age_actuel(t)) + self.MaintienCh.prob_passage_ch_ch(self.age_actuel(t), self.duree_sinistre(t)) * self.effectifs_par_etat(t-1)["ch"]
            df["inv"] = self.effectifs_par_etat( t-1)["inv"] + self.prob_passage_inc_inv(self.age_actuel(t), self.duree_sinistre(t)) * self.effectifs_par_etat( t-1)["inc"] + self.Incidence.prob_entree_inval(self.age_actuel(t)) * self.effectifs_par_etat( t-1)["v"]

            df["inc"] = self.MaintienIncap.prob_passage_inc_inc(self.age_actuel(t), self.duree_sinistre(t)) * self.effectifs_par_etat( t-1)["inc"] + self.Incidence.prob_entree_incap(self.age_actuel(t)) / 12 * self.effectifs_par_etat( t-1)["v"]

            df["dc"] = self.effectifs_par_etat( t-1)["dc"] + self.Mortalite.prob_dc(self.sexe(), self.age_actuel(t)) / 12 * (self.effectifs_par_etat( t-1)["inc"]
                                                                                        + self.effectifs_par_etat( t-1)["inv"]
                                                                                        + self.effectifs_par_etat( t-1)["ch"]
                                                                                        + self.effectifs_par_etat( t-1)["v"])

            df["lps"] = self.effectifs_par_etat( t-1)["lps"] + self.Lapse.prob_rachat(self.produit(), self.anciennete_contrat_mois(t)) * self.effectifs_par_etat(self, t-1)["v"]
            df['v'] = self.effectifs_par_etat(self, t-1)["v"] - df['lps'] - df['dc'] - df['inc'] - df['inv'] - df['ch']

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
            som1 = som1 + ((1 + taux) ^ -(i / 12))* ADEFlux.PassageInval.nombre_passage_inval(agentree, durecoulee + i) * prov
            som2 = som2 + ((1 + taux) ^ -((i + 1) / 12)) * ADEFlux.PassageInval.nombre_passage_inval(agentree, durecoulee + i + 1) * prov
        return ((som1 + som2) / (2 * l))

    def amortisation_schedule(amount, annualinterestrate, paymentsperyear, years):
        df = pd.DataFrame({'PrincipalPaid' :[np.ppmt(annualinterestrate/paymentsperyear, i+1, paymentsperyear*years, amount) for i in range(paymentsperyear*years)],
                           'InterestPaid' :[np.ipmt(annualinterestrate/paymentsperyear, i+1, paymentsperyear*years, amount) for i in range(paymentsperyear*years)]})
        df['Instalment'] = df.PrincipalPaid + df.InterestPaid
        df['CumulativePrincipal'] = np.cumsum(df.PrincipalPaid)
        df['Principal'] = amount
        df['Balance'] = df['Principal'] + df['CumulativePrincipal']
        return (df)

#data_files_path ='C:/Users/work/OneDrive/modele_emprunteur/CSV'
ModelPoint = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MODEL_POINT.csv', sep=";")
projection = ADEFlux(ModelPoint.loc[[0]])
projection.produit()