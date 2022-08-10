# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:26:37 2022

@author: work
"""

import pandas as pd

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

    Incidence = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//INCIDENCE.csv', sep=";")
    Lapse = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//LAPSE.csv', sep=";")
    MaintienCh = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MAINTIEN_CH.csv', sep=";")
    MaintienIncap = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MAINTIEN_INCAP.csv', sep=";")
    MortaliteTH = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MORTALITE_TF0002.csv', sep=";")
    MortaliteTF = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MORTALITE_TH0002.csv', sep=";")
    PassageInval = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//PASSAGE_INVAL.csv', sep=";")

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

    def nb_contrats(self,t):
        return self.ModelPointRow.nb_contrats(t-1)

    def duree_sinistre(self,t):
        return self.ModelPointRow.duree_sinistre+t

    def max_age_mortality_th():
        return max(ADEFlux.MortaliteTH["age_x"])

    def max_age_mortality_tf():
        return max(ADEFlux.MortaliteTF["age_x"])

    def max_age_incidence():
        return max(ADEFlux.Incidence["age_x"])

    def prob_dc(self,t):
        if self.sexe()=='F':
            if self.age_actuel(t)>=self.max_age_mortality_tf():
                return ADEFlux.MortaliteTF.loc[ADEFlux.MortaliteTF["age_x"]==self.max_age_mortality_tf(), "Qx"]
            else :
                return ADEFlux.MortaliteTF.loc[ADEFlux.MortaliteTF["age_x"]==self.age_actuel(t), "Qx"]
        else:
            if self.age_actuel(t)>=self.max_age_mortality_th():
                return ADEFlux.MortaliteTH.loc[ADEFlux.MortaliteTH["age_x"]==self.max_age_mortality_th(), "Qx"]
            else :
                return ADEFlux.MortaliteTH.loc[ADEFlux.MortaliteTH["age_x"]==self.age_actuel(t), "Qx"]

    def prob_entree_incap(self, t):
        max__age__incidence = self.max_age_incidence()
        age_actuel__ = self.age_actuel(t)
        if age_actuel__<max__age__incidence:
            return ADEFlux.Incidence.loc[ADEFlux.Incidence["age_x"]==age_actuel__, "Incidence_en_incap"]
        else:
            return ADEFlux.Incidence.loc[ADEFlux.Incidence["age_x"]==max__age__incidence, "Incidence_en_incap"]

    def prob_entree_chomage(self, t):
        max__age__incidence = self.max_age_incidence()
        age_actuel__ = self.age_actuel(t)
        if age_actuel__<max__age__incidence:
            return ADEFlux.Incidence.loc[ADEFlux.Incidence["age_x"]==age_actuel__, "Incidence_en_chomage"]
        else:
            return ADEFlux.Incidence.loc[ADEFlux.Incidence["age_x"]==max__age__incidence, "Incidence_en_chomage"]

    def prob_entree_inval(self, t):
        max__age__incidence = self.max_age_incidence()
        age_actuel__ = self.age_actuel(t)
        if age_actuel__<max__age__incidence:
            return ADEFlux.Incidence.loc[ADEFlux.Incidence["age_x"]==age_actuel__, "Incidence_en_inval"]
        else:
            return ADEFlux.Incidence.loc[ADEFlux.Incidence["age_x"]==max__age__incidence, "Incidence_en_inval"]

    def prob_rachat(self,t):
        return ADEFlux.Lapse.loc[ADEFlux.Lapse["produit"]==self.produit(), str(self.anciennete_contrat_mois(t))]

    def nombre_maintien_chomage(self, t, age, anciennete_chomage):
        duree_sinistre__ = self.duree_sinistre(t)
        age_actuel__ = self.age_actuel(t)
        if(round(age_actuel__-duree_sinistre__)<=46):
            if(duree_sinistre__<=34):
                return ADEFlux.MaintienCh.loc[ADEFlux.MaintienCh["Age_Anciennete"]==age,str(anciennete_chomage)]
            else:
                return ADEFlux.MaintienCh.loc[ADEFlux.MaintienCh["Age_Anciennete"]==age,"35"]
        else:
            if (duree_sinistre__<=34):
                return ADEFlux.MaintienCh.loc[ADEFlux.MaintienCh["Age_Anciennete"]==46,str(anciennete_chomage)]
            else:
                return ADEFlux.MaintienCh.loc[ADEFlux.MaintienCh["Age_Anciennete"]==46,"35"]

    def nombre_maintien_incap(self, t, age, anciennete_incap):
        duree_sinistre__ = self.duree_sinistre(t)
        age_actuel__ = self.age_actuel(t)
        if(round(age_actuel__-self.duree_sinistre(t))<=46):
            if(duree_sinistre__<=34):
                return ADEFlux.MaintienIncap.loc[ADEFlux.MaintienIncap["Age_Anciennete"]==round(age_actuel__-duree_sinistre__),str(duree_sinistre__+1)]
            else:
                return ADEFlux.MaintienIncap.loc[ADEFlux.MaintienIncap["Age_Anciennete"]==round(age_actuel__-duree_sinistre__),"35"]
        else:
            if (self.duree_sinistre(t)<=34):
                return ADEFlux.MaintienIncap.loc[ADEFlux.MaintienIncap["Age_Anciennete"]==46,str(duree_sinistre__+1)]
            else:
                return ADEFlux.MaintienIncap.loc[ADEFlux.MaintienIncap["Age_Anciennete"]==46,"35"]

    def nombre_passage_inval(self, t, age, anciennete_inval):
        duree_sinistre__ = self.duree_sinistre(t)
        age_actuel__ = self.age_actuel(t)
        if(round(age_actuel__-duree_sinistre__)<=43):
            if(duree_sinistre__<=34):
                return ADEFlux.PassageInval.loc[ADEFlux.PassageInval["Age_Anciennete"]==round(age_actuel__-duree_sinistre__),str(duree_sinistre__+1)]
            else:
                return ADEFlux.PassageInval.loc[ADEFlux.PassageInval["Age_Anciennete"]==round(age_actuel__-duree_sinistre__),"35"]
        else:
            if (duree_sinistre__<=34):
                return ADEFlux.PassageInval.loc[ADEFlux.PassageInval["Age_Anciennete"]==43,str(duree_sinistre__+1)]
            else:
                return ADEFlux.PassageInval.loc[ADEFlux.PassageInval["Age_Anciennete"]==43,"35"]

    def pmxcho(self, agentree, durecoulee, D1, D2, taux):
        som1 = 0
        som2 = 0
        l = self.nombre_maintien_chomage(agentree, durecoulee)
        for i in range(D1,(D2-1)):
            som1 = som1 + ((1 + taux)^(-(i / 12))) * self.nombre_maintien_chomage(agentree, durecoulee + i)
            som2 = som2 + ((1 + taux)^(-((i + 1) / 12))) * self.nombre_maintien_chomage(agentree, durecoulee + i+1)
        return((som1 + som2) / (2 * l))

class LoiMaintienChomage():
    """
        Represente la loi de maintien en chômage
    """
    def __init__(self, maintienCh) -> None:
        self.MaintienCh = maintienCh

    def nombre_maintien_chomage(self, age_entre, anciennete_chomage):
        """

        Parameters
        ----------
        age : int
            age d'entrée en chômage.
        anciennete_chomage : int
            duration en état de chômage.

        Returns
        -------
        TYPE
            Nombre d'indiv en situation de chômage selon l'âge et la duration.

        """
        if(age_entre<=46):
            if (anciennete_chomage<=34):
                return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==age_entre,str(anciennete_chomage)]
            else:
                return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==age_entre,"35"]
        else:
            if (anciennete_chomage<=34):
                return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==46,str(anciennete_chomage)]
            else:
                return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==46,"35"]

class LoiMaintienIncapacite():
    """
        Represente la loi de maintien en chômage
    """
    def __init__(self, maintienIncap) -> None:
        self.MaintienIncap = maintienIncap

    def nombre_maintien_incap(self, age_entre, anciennete_incap):
        if(age_entre<=46):
            if (anciennete_incap<=34):
                return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==age_entre,str(anciennete_incap)]
            else:
                return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==age_entre,"35"]
        else:
            if (anciennete_incap<=34):
                return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==46,str(anciennete_incap)]
            else:
                return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==46,"35"]

class LoiMaintienInvalidite():
    """
        Represente la loi de maintien en chômage
    """
    def __init__(self, passageInval) -> None:
        self.PassageInval = passageInval

    def nombre_passage_inval(self, age_entree, anciennete_incap):
        if (age_entree<=43):
            if (anciennete_incap<=34):
                return self.PassageInval.loc[self.PassageInval["Age_Anciennete"]==age_entree,str(anciennete_incap)]
            else:
                return self.PassageInval.loc[self.PassageInval["Age_Anciennete"]==age_entree,"35"]
        else:
            if (anciennete_incap<=34):
                return self.PassageInval.loc[self.PassageInval["Age_Anciennete"]==43,str(anciennete_incap)]
            else:
                return self.PassageInval.loc[self.PassageInval["Age_Anciennete"]==43,"35"]

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


#data_files_path ='C:/Users/work/OneDrive/modele_emprunteur/CSV'
Incidence = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//INCIDENCE.csv', sep=";")
Lapse = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//LAPSE.csv', sep=";")
MaintienCh = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MAINTIEN_CH.csv', sep=";")
MaintienIncap = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MAINTIEN_INCAP.csv', sep=";")
MortaliteTH = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MORTALITE_TF0002.csv', sep=";")
MortaliteTF = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MORTALITE_TH0002.csv', sep=";")
PassageInval = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//PASSAGE_INVAL.csv', sep=";")

ModelPoint = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MODEL_POINT.csv', sep=";")
projection = ADEFlux(ModelPoint.loc[[0]])
projection.produit()