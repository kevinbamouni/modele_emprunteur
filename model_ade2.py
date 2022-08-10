# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:26:37 2022

@author: work
"""

import pandas as pd
import numpy as np

class MemoizeClass:
    """
     Decorateur de Class pour implementation de Memoization : https://python-course.eu/advanced-python/memoization-decorators.php
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}
    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.fn(*args)
        return self.memo[args]

def MemoizeFunction(f):
    """
    Decorateur de Fonction pour implementation de Memoization : https://python-course.eu/advanced-python/memoization-decorators.php

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
    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper

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

    # def nombre_maintien_chomage(self, age_entre, anciennete_chomage):
    #     """

    #     Parameters
    #     ----------
    #     age : int
    #         age d'entrée en chômage.
    #     anciennete_chomage : int
    #         duration en état de chômage.

    #     Returns
    #     -------
    #     TYPE
    #         Nombre d'indiv en situation de chômage selon l'âge et la duration.

    #     """
    #     if(age_entre<=46):
    #         if (anciennete_chomage<=34):
    #             return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==age_entre,str(anciennete_chomage)]
    #         else:
    #             return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==age_entre,"35"]
    #     else:
    #         if (anciennete_chomage<=34):
    #             return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==46,str(anciennete_chomage)]
    #         else:
    #             return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==46,"35"]

    def  nombre_maintien_chomage(self, age_entre, anciennete_chomage):
        return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==age_entre,str(max(anciennete_chomage, self.MaintienCh.shape[1]-1))]

    # def prob_passage_ch_ch(self, age, anciennete_chomage):
    #     if (anciennete_chomage < self.MaintienCh.shape[1]-1):
    #         if (age<=65):
    #           self.nombre_maintien_chomage(age, anciennete_chomage+1)/self.nombre_maintien_chomage(age, anciennete_chomage)
    #         else:
    #           self.nombre_maintien_chomage(65, anciennete_chomage+1)/self.nombre_maintien_chomage(65, anciennete_chomage)
    #     else:
    #         if(age<=65):
    #             self.nombre_maintien_chomage(age, 35)/self.nombre_maintien_chomage(age, 34)
    #         else:
    #             self.nombre_maintien_chomage(65, 35)/self.nombre_maintien_chomage(65, 34)
    def prob_passage_ch_ch(self, age, anciennete_chomage):
        return self.nombre_maintien_chomage(age, anciennete_chomage+1)/self.nombre_maintien_chomage(age, anciennete_chomage)

class LoiMaintienIncapacite():
    """
        Represente la loi de maintien en chômage
    """
    def __init__(self, maintienIncap) -> None:
        self.MaintienIncap = maintienIncap

    # def nombre_maintien_incap(self, age_entre, anciennete_incap):
    #     if(age_entre<=46):
    #         if (anciennete_incap<=34):
    #             return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==age_entre,str(anciennete_incap)]
    #         else:
    #             return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==age_entre,"35"]
    #     else:
    #         if (anciennete_incap<=34):
    #             return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==46,str(anciennete_incap)]
    #         else:
    #             return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==46,"35"]

    def nombre_maintien_incap(self, age_entre, anciennete_incap):
        return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==age_entre,str(max(anciennete_incap, self.MaintienIncap.shape[1]-1))]

    # def prob_passage_inc_inc(self, age, anciennete_inc):
    #     if (anciennete_inc < self.MaintienIncap.shape[1]-1):
    #         if (age<=65):
    #           self.nombre_maintien_incap(age, anciennete_inc+1)/self.nombre_maintien_incap(age, anciennete_inc)
    #         else:
    #           self.nombre_maintien_incap(65, anciennete_inc+1)/self.nombre_maintien_incap(65, anciennete_inc)
    #     else:
    #         if(age<=65):
    #             self.nombre_maintien_incap(age, 35)/self.nombre_maintien_incap(age, 34)
    #         else:
    #             self.nombre_maintien_incap(65, 35)/self.nombre_maintien_incap(65, 34)

    def prob_passage_inc_inc(self, age, anciennete_inc):
        return self.nombre_maintien_incap(age, anciennete_inc+1)/self.nombre_maintien_incap(age, anciennete_inc)

class LoiPasssageInvalidite():
    """
        Represente la loi de passage en invalidité
    """
    def __init__(self, passageInval) -> None:
        self.PassageInval = passageInval

    def nombre_passage_inval(self, age_entree, anciennete_incap):
        return self.PassageInval.loc[self.PassageInval["Age_Anciennete"]==age_entree,str(max(anciennete_incap, self.PassageInval.shape[1]-1))]

    # def nombre_passage_inval(self, age_entree, anciennete_incap):
    #     if (age_entree<=43):
    #         if (anciennete_incap<=34):
    #             return self.PassageInval.loc[self.PassageInval["Age_Anciennete"]==age_entree,str(anciennete_incap+1)]
    #         else:
    #             return self.PassageInval.loc[self.PassageInval["Age_Anciennete"]==age_entree,"35"]
    #     else:
    #         if (anciennete_incap<=34):
    #             return self.PassageInval.loc[self.PassageInval["Age_Anciennete"]==43,str(anciennete_incap)]
    #         else:
    #             return self.PassageInval.loc[self.PassageInval["Age_Anciennete"]==43,"35"]

    def prob_passage_inc_inv(self, age, anciennete_inc):
        return (self.nombre_passage_inval(age, anciennete_inc+1) - self.nombre_passage_inval(age, anciennete_inc))/self.nombre_passage_inval(age, anciennete_inc)


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
        if age_actuel<self.max_age_incidence():
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
    Mortalite = LoiMortalite(MortaliteTHData)
    PassageInval = LoiPasssageInvalidite(PassageInvalData)
    ReferentielProduit = TblProd(referentielProduit)

    def mp_id(self):
        return self.ModelPointRow['mp_id']

    def sexe(self):
        return self.ModelPointRow.sexe

    def etat(self):
        return self.ModelPointRow.etat

    def age_actuel(self,t):
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

    def duree_restante(self,t):
        if t==0:
            return self.ModelPointRow.duree_restante
        else:
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
        return self.ModelPointRow.duree_sinistre+t

    def effectifs_par_etat(self, t):
        if t==0:
            return pd.DataFrame({'v': [lambda: self.nb_contrats if self.etat() == 'v' else 0],
                                 'ch': [lambda: self.nb_contrats if self.etat() == 'ch' else 0],
                                 'inv': [lambda: self.nb_contrats if self.etat() == 'inv' else 0],
                                 'inc': [lambda: self.nb_contrats if self.etat() == 'inc' else 0],
                                 'dc': [0],
                                 'lps': [0]
                                 })
        else:
            pd.DataFrame({'v': [lambda: self.nb_contrats if self.etat() == 'v' else 0],
                          'ch': [lambda: self.nb_contrats if self.etat() == 'ch' else 0],
                          'inv': [lambda: self.nb_contrats if self.etat() == 'inv' else 0],
                          'inc': [lambda: self.nb_contrats if self.etat() == 'inc' else 0],
                          'dc': self.Mortalite.prob_dc(self.sexe(), self.age_actuel(t))
                          * (self.effectifs_par_etat(t-1)["inc"]
                             + self.effectifs_par_etat(t-1)["inv"]
                             + self.effectifs_par_etat(t-1)["ch"]),
                          'lps': [0]
                          })
            self.effectifs_par_etat(t-1)

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