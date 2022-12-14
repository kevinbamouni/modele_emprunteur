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
from datetime import datetime


class TblProd():
    def __init__(self, tblProduct) -> None:
        self.TblProduct = tblProduct

    @functools.cache
    def get_tx_prime_dc(self, productCode):
        return self.TblProduct.loc[self.TblProduct["produits"]==productCode,"tx_prime_dc"].values[0]

    @functools.cache
    def get_tx_prime_inc(self, productCode):
        return self.TblProduct.loc[self.TblProduct["produits"]==productCode,"tx_prime_inc"].values[0]

    @functools.cache
    def get_tx_prime_chomage(self, productCode):
        return self.TblProduct.loc[self.TblProduct["produits"]==productCode,"tx_prime_chomage"].values[0]

    @functools.cache
    def get_tx_frais_admin(self, productCode):
        return self.TblProduct.loc[self.TblProduct["produits"]==productCode,"tx_frais_admin"].values[0]

    @functools.cache
    def get_tx_frais_acq(self, productCode):
        return self.TblProduct.loc[self.TblProduct["produits"]==productCode,"tx_frais_acq"].values[0]

    @functools.cache
    def get_tx_comm(self, productCode):
        return self.TblProduct.loc[self.TblProduct["produits"]==productCode,"tx_comm"].values[0]

    @functools.cache
    def get_tx_profit_sharing_assureur(self, productCode):
        return self.TblProduct.loc[self.TblProduct["produits"]==productCode,"tx_profit_sharing_assureur"].values[0]

    @functools.cache
    def get_tx_profit_sharing_partenaire(self, productCode):
        return self.TblProduct.loc[self.TblProduct["produits"]==productCode,"tx_profit_sharing_partenaire"].values[0]

    @functools.cache
    def get_tx_production_financiere(self, productCode):
        return self.TblProduct.loc[self.TblProduct["produits"]==productCode,"tx_production_financiere"].values[0]
    
    @functools.cache
    def get_tx_frais_gest_sin(self, productCode):
        return self.TblProduct.loc[self.TblProduct["produits"]==productCode,"tx_frais_gest_sin"].values[0]

class LoiMaintienChomage():
    """
        Represente la loi de maintien en ch??mage entre t et t+1
    """
    def __init__(self, maintienCh) -> None:
        self.MaintienCh = maintienCh
    
    @functools.cache
    def max_age_pass_ch(self):
        """age max de la loi d'incidence

        Returns:
            int: age max de la loi d'incidence
        """
        return max(self.MaintienCh["Age_Anciennete"])
    
    @functools.cache
    def max_duration_pass_ch(self):
        return self.MaintienCh.shape[1]-2

    @functools.cache
    def  nombre_maintien_chomage(self, age_entree, anciennete_chomage):
        """_summary_

        Args:
            age_entree (int): age entree en chomage
            anciennete_chomage (int): duration en chomage

        Returns:
            int: nombre en chomage
        """
        if age_entree<=self.max_age_pass_ch() and (anciennete_chomage+1)<=self.max_duration_pass_ch():
            try:
                return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==math.floor(age_entree), str(anciennete_chomage)].values[0]
            except KeyError:
                return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==math.floor(age_entree), str(self.MaintienCh.shape[1]-2)].values[0]
        else: return self.MaintienCh.loc[self.MaintienCh["Age_Anciennete"]==self.max_age_pass_ch(), str(self.max_duration_pass_ch())].values[0]
    
    @functools.cache
    def prob_passage_ch_ch(self, age, anciennete_chomage):
        """proba de rester en chomage

        Args:
            age (int): age entre chomage
            anciennete_chomage (int): duration en chomage

        Returns:
            decimal: proba rester en chomage
        """
        if age<=self.max_age_pass_ch() and (anciennete_chomage+1)<=self.max_duration_pass_ch():
            return self.nombre_maintien_chomage(age, anciennete_chomage+1)/self.nombre_maintien_chomage(age, anciennete_chomage)
        else: return 0

class LoiMaintienIncapacite():
    """
        Represente la loi de maintien en incapacite entre t et t+1
    """
    def __init__(self, maintienIncap) -> None:
        self.MaintienIncap = maintienIncap
    
    @functools.cache
    def max_age_pass_inc(self):
        """age max de la loi d'incidence

        Returns:
            int: age max de la loi d'incidence
        """
        return max(self.MaintienIncap["Age_Anciennete"])
    
    @functools.cache
    def max_duration_pass_inc(self):
        return self.MaintienIncap.shape[1]-2

    @functools.cache
    def nombre_maintien_incap(self, age_entree, anciennete_incap):
        """_summary_

        Args:
            age_entree (int): age entree en incap
            anciennete_incap (int): duraiton en incap

        Returns:
            int: effectif en incap
        """
        if age_entree>self.max_age_pass_inc() or (anciennete_incap+1)>self.max_duration_pass_inc():
            return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==self.max_age_pass_inc(),str(self.max_duration_pass_inc())].values[0]
        else:
            try:
                return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==math.floor(age_entree),str(anciennete_incap)].values[0]
            except KeyError:
                return self.MaintienIncap.loc[self.MaintienIncap["Age_Anciennete"]==math.floor(age_entree),str(self.MaintienIncap.shape[1]-2)].values[0]

    @functools.cache
    def prob_passage_inc_inc(self, age, anciennete_inc):
        """proba de rester en incap

        Args:
            age (int): age entree en incap
            anciennete_inc (int): duration en incap

        Returns:
            decimal: proba de rester en incap
        """
        if age>self.max_age_pass_inc() or (anciennete_inc+1)>self.max_duration_pass_inc():
            return 0
        else :return self.nombre_maintien_incap(age, anciennete_inc+1)/self.nombre_maintien_incap(age, anciennete_inc)

class LoiPasssageInvalidite():
    """
        Represente la loi de passage en invalidit?? entre t et t+1
    """
    def __init__(self, passageInval) -> None:
        self.PassageInval = passageInval
    
    @functools.cache
    def max_age_pass_inval(self):
        """age max de la loi d'incidence

        Returns:
            int: age max de la loi d'incidence
        """
        return max(self.PassageInval["Age_Anciennete"])
    
    @functools.cache
    def max_duration_pass_inval(self):
        return self.PassageInval.shape[1]-2

    @functools.cache
    def nombre_passage_inval(self, age_entree, anciennete_incap):
        """_summary_

        Args:
            age_entree (int): age entree en inval
            anciennete_incap (int): duration en inval

        Returns:
            int: nombre en inval
        """
        try:
            return self.PassageInval.loc[self.PassageInval["Age_Anciennete"]==math.floor(age_entree),str(anciennete_incap)].values[0]
        except KeyError:
            return self.PassageInval.loc[self.PassageInval["Age_Anciennete"]==math.floor(age_entree),str(self.PassageInval.shape[1]-2)].values[0]

class LoiIncidence():
    """Probabilit?? des valides de passer en incap, ou inval, ou chomage, aussi appel??e loi d'incidence
    """
    def __init__(self, incidence) -> None:
        self.Incidence = incidence

    @functools.cache
    def max_age_incidence(self):
        """age max de la loi d'incidence

        Returns:
            int: age max de la loi d'incidence
        """
        return max(self.Incidence["age_x"])

    @functools.cache
    def prob_entree_incap(self, age_actuel):
        """proba pour un valide d'entr?? en incap

        Args:
            age_actuel (int): age actuel

        Returns:
            flocat: proba pour un valide d'entr?? en incap
        """
        if age_actuel<self.max_age_incidence():
            return self.Incidence.loc[self.Incidence["age_x"]==math.floor(age_actuel), "Incidence_en_incap"].values[0]
        else:
            return self.Incidence.loc[self.Incidence["age_x"]==self.max_age_incidence(), "Incidence_en_incap"].values[0]

    @functools.cache
    def prob_entree_chomage(self, age_actuel):
        """proba pour un valide d'entr?? en chomage

        Args:
            age_actuel (int): age actuel

        Returns:
            flocat: proba pour un valide d'entr?? en chomage
        """
        if age_actuel < self.max_age_incidence():
            return self.Incidence.loc[self.Incidence["age_x"]==math.floor(age_actuel), "Incidence_en_chomage"].values[0]
        else:
            return self.Incidence.loc[self.Incidence["age_x"]==self.max_age_incidence(), "Incidence_en_chomage"].values[0]

    @functools.cache
    def prob_entree_inval(self, age_actuel):
        """proba pour un valide d'entr?? en inval

        Args:
            age_actuel (int): age actuel

        Returns:
            flocat: proba pour un valide d'entr?? en inval
        """
        if age_actuel<self.max_age_incidence():
            return self.Incidence.loc[self.Incidence["age_x"]==math.floor(age_actuel), "Incidence_en_inval"].values[0]
        else:
            return self.Incidence.loc[self.Incidence["age_x"]==self.max_age_incidence(), "Incidence_en_inval"].values[0]

class LoiMortalite():
    """Loi de mortalit?? ou table de mortalit??, probabilit??s pour un individu d'age x de decede avant l'age x+1
    """
    def __init__(self, mortaliteTH, mortaliteTF, tech_int_rate=0) -> None:
        self.MortaliteTH = mortaliteTH
        self.MortaliteTF = mortaliteTF
        self.tech_int_rate = tech_int_rate

    @functools.cache
    def max_age_mortality_th(self):
        """max age mortalite TH

        Returns:
            int: max age mortalite TH
        """
        return max(self.MortaliteTH["age_x"])
    
    @functools.cache
    def max_age_mortality_tf(self):
        """max age mortalite TF

        Returns:
            int: max age mortalite TF
        """
        return max(self.MortaliteTF["age_x"])

    @functools.cache
    def prob_dc(self, sexe, age_actuel):
        """Return Qx for age and sex.

        Args:
            sexe (_type_): _description_
            age_actuel (_type_): _description_

        Returns:
            _type_: _description_
        """
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
    
    @functools.cache
    def lx(self, sexe, age_actuel):
        """The number of persons remaining at age

        Args:
            sexe (_type_): _description_
            age_actuel (_type_): _description_

        Returns:
            _type_: _description_
        """
        if age_actuel == 0:
            return 100000
        else:
            return self.lx(sexe, age_actuel-1) - self.dx(sexe, age_actuel-1)
    
    @functools.cache
    def dx(self, sexe, age_actuel):
        """The number of persons who die between ages ``x`` and ``x+1``

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.lx(sexe, age_actuel) * self.prob_dc(sexe, age_actuel)
    
    def disc(self):
        """_summary_

        Args:
            discountrate (_type_): _description_

        Returns:
            _type_: _description_
        """
        return 1 / (1 + self.tech_int_rate)
    
    @functools.cache
    def Dx(self, sexe, age_actuel):
        """The commutation column :math:`D_{x} = l_{x}v^{x}`.

        Args:
            sexe (_type_): _description_
            age_actuel (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.lx(sexe, age_actuel) * pow(self.disc(), age_actuel)
    
    @functools.cache
    def Nx(self, sexe, age_actuel):
        """ The commutation column :math:`N_x`. 

        Args:
            sexe (_type_): _description_
            age_actuel (_type_): _description_

        Returns:
            _type_: _description_
        """
        if age_actuel >= 110:    # TODO: Get the last age from the table
            return self.Dx(sexe, age_actuel)
        else:
            return self.Nx(sexe, age_actuel+1) + self.Dx(sexe, age_actuel)
    
    @functools.cache
    def Mx(self, sexe, age_actuel):
        """The commutation column :math:`M_x`.

        Args:
            sexe (_type_): _description_
            age_actuel (_type_): _description_

        Returns:
            _type_: _description_
        """
        if age_actuel >= 110:
            return self.Dx(sexe, age_actuel)
        else:
            return self.Mx(sexe, age_actuel+1) + self.Cx(sexe, age_actuel)
        
    def Exn(self, sexe, age_actuel, n):
        """The value of an endowment on a person at age ``x``
        payable after n years

        .. math::

        {}_{n}E_x

        Args:
            sexe (_type_): _description_
            age_actuel (_type_): _description_
            n (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.Dx(sexe, age_actuel) == 0:
            return 0
        else:
            return self.Dx(sexe, age_actuel+n) / self.Dx(sexe, age_actuel)
        
    def Cx(self, sexe, age_actuel):
        """The commutation column :math:`\\overline{C_x}`.

        Args:
            sexe (_type_): _description_
            age_actuel (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.dx(sexe, age_actuel) * pow(self.disc(), (age_actuel+0.5))
    
    def Axn(self, sexe, age_actuel, n, f=0):
        """The present value of an assurance on a person at age ``x`` payable
            immediately upon death, optionally with an waiting period of ``f`` years.

            .. math::

                \\require{enclose}{}_{f|}\\overline{A}^{1}_{x:\\enclose{actuarial}{n}}

        Args:
            sexe (_type_): _description_
            age_actuel (_type_): _description_
            n (_type_): _description_
            f (int, optional): waiting period in years. Defaults to 0.

        Returns:
            _type_: _description_
        """
        if self.Dx(sexe, age_actuel) == 0:
            return 0
        else:
            return (self.Mx(sexe, age_actuel+f) - self.Mx(sexe, age_actuel+f+n)) / self.Dx(sexe, age_actuel)
        
    def Ax(self, sexe, age_actuel, f=0):
        """The present value of a lifetime assurance on a person at age ``x``
            payable immediately upon death, optionally with an waiting period of ``f`` years.

            .. math::

        \\require{enclose}{}_{f|}\\overline{A}_{x}

        Args:
            sexe (_type_): _description_
            age_actuel (_type_): _description_
            f (int, optional): waiting period in years. Defaults to 0.

        Returns:
            _type_: _description_
        """
        if self.Dx(sexe, age_actuel) == 0:
            return 0
        else:
            return self.Mx(sexe,age_actuel+f) / self.Dx(sexe, age_actuel)
        
    def AnnDuex(self, sexe, age_actuel, k, f=0):
        """The present value of a lifetime annuity due.

        Args:
            sexe (_type_): _description_
            age_actuel (_type_): _description_
            k (int): number of split payments in a year
            f (int, optional): waiting period in years. Defaults to 0.

        Returns:
            _type_: _description_
        """
        if self.Dx(sexe, age_actuel) == 0:
            return 0
        result = (self.Nx(sexe, age_actuel+f)) / self.Dx(sexe, age_actuel)
        if k > 1:
            return result - (k-1) / (2*k)
        else:
            return result
        
    def AnnDuenx(self, sexe, age_actuel, n , k=1, f=0):
        """The present value of an annuity-due.

            .. math::

                \\require{enclose}{}_{f|}\\ddot{a}_{x:\\enclose{actuarial}{n}}^{(k)}
        Args:
            sexe (_type_): sexe
            age_actuel (_type_): age
            n (_type_): length of annuity payments in years
            k (int, optional): number of split payments in a year Defaults to 1.
            f (int, optional): waiting period in years Defaults to 0.

        Returns:
            _type_: _description_
        """
        if self.Dx(sexe, age_actuel) == 0:
            return 0
        result = (self.Nx(sexe, age_actuel+f) - self.Nx(sexe, age_actuel+f+n)) / self.Dx(sexe, age_actuel)

        if k > 1:
            return result - (k-1) / (2*k) * (1 - self.Dx(sexe, age_actuel+f+n) / self.Dx(sexe, age_actuel))
        else:
            return result

class LoiRachat():
    def __init__(self, lapse) -> None:
        self.Lapse = lapse

    @functools.cache
    def prob_rachat(self, produit, anciennete_contrat_mois):
        """probabilit?? de rachat

        Args:
            produit (char): product name or product code
            anciennete_contrat_mois (int): duration du contrat en mois

        Returns:
            decimal: probabilit?? de rachat du contrat
        """
        try:
            return self.Lapse.loc[self.Lapse["produit"]==produit, str(anciennete_contrat_mois)].values[0]
        except KeyError:
            return self.Lapse.loc[self.Lapse["produit"]==produit, str(self.Lapse.shape[1]-2)].values[0]
    
class TauxForward():
    def __init__(self, tauxforward) -> None:
        self.tauxforward = tauxforward
    
    @functools.cache
    def max_time(self):
        return max(self.tauxforward["TIME"])
    
    @functools.cache
    def get_taux_forward_at_t(self, t):
        if t>self.max_time() or t<=0:
            return 0
        else : return self.tauxforward.loc[self.tauxforward["TIME"]==t, 'TX_FORWARD'].values[0]
    
    @functools.cache
    def fact_actu(self, t):
        return pow((1+self.get_taux_forward_at_t(t)), -t)
            
        
        
class ADECashFLowModel():
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
    
    ModelConfig = {}
    Incidence = LoiIncidence(pd.DataFrame())
    Lapse = LoiRachat(pd.DataFrame())
    MaintienCh = LoiMaintienChomage(pd.DataFrame())
    MaintienIncap = LoiMaintienIncapacite(pd.DataFrame())
    Mortalite = LoiMortalite(pd.DataFrame(), pd.DataFrame())
    PassageInval = LoiPasssageInvalidite(pd.DataFrame())
    ReferentielProduit = TblProd(pd.DataFrame())
    ForwardRate = TauxForward(pd.DataFrame())
    
    @classmethod
    def config(cls, inputPathDict):
        """Configure the model inputs :
            dict = {'input 1': 'path_to_input1',
                    'input 2': 'path_to_input2',
                    ...}

        Args:
            inputPathDict (dict): configuration information
        """
        cls.ModelConfig = inputPathDict
        cls.Incidence = LoiIncidence(pd.read_csv(inputPathDict['IncidenceData'], sep=";"))
        cls.Lapse = LoiRachat(pd.read_csv(inputPathDict['LapseData'], sep=";"))
        cls.MaintienCh = LoiMaintienChomage(pd.read_csv(inputPathDict['MaintienChData'], sep=";"))
        cls.MaintienIncap = LoiMaintienIncapacite(pd.read_csv(inputPathDict['MaintienIncapData'], sep=";"))
        cls.Mortalite = LoiMortalite(pd.read_csv(inputPathDict['MortaliteTHData'], sep=";"), pd.read_csv(inputPathDict['MortaliteTFData'], sep=";"))
        cls.PassageInval = LoiPasssageInvalidite(pd.read_csv(inputPathDict['PassageInvalData'], sep=";"))
        cls.ReferentielProduit = TblProd(pd.read_csv(inputPathDict['referentielProduit'], sep=";"))
        cls.ForwardRate = TauxForward(pd.read_csv(inputPathDict['TxForward'], sep=";"))
    
    @classmethod
    def printconfig(cls):
        """Print the model configuration
        """
        for q, a in cls.ModelConfig.items():
            print(' {0} :  {1}.'.format(q, a))

    @functools.cache
    def mp_id(self):
        return self.ModelPointRow.mp_id

    @functools.cache
    def sexe(self):
        return self.ModelPointRow.sexe

    @functools.cache
    def etat(self):
        return self.ModelPointRow.etat

    @functools.cache
    def age_actuel(self, t):
        return math.floor(self.ModelPointRow.age_actuel + t/12)

    @functools.cache
    def age_souscription_annee(self):
        return self.ModelPointRow.age_souscription_annee

    @functools.cache
    def annee_souscription(self):
        return self.ModelPointRow.annee_souscription

    @functools.cache
    def anciennete_contrat_annee(self,t):
        return self.ModelPointRow.anciennete_contrat_annee + t/12

    @functools.cache
    def anciennete_contrat_mois(self,t):
        return self.ModelPointRow.anciennete_contrat_mois + t

    @functools.cache
    def duree_pret(self):
        return self.ModelPointRow.duree_pret

    @functools.cache
    def age_fin(self):
        return self.ModelPointRow.age_fin

    @functools.cache
    def ci(self):
        """
            retourne le Capital Initial du pr??t
        Returns:
            decimal: Capital initial du pr??t
        """
        return self.ModelPointRow.ci

    @functools.cache
    def crd(self, t):
        """
            retourne le Montant Restant d?? du cr??dit juste apr??s la mensualit?? vers?? ?? l'instant t
        Returns:
            decimal: montant du capital restant d??
        """
        crd = self.loan_balance_at_n(self.ci(), self.taux_nominal(), 12, self.duree_pret()/12, (self.duree_pret()-self.duree_restante(t)))
        return crd
    
    @functools.cache
    def duree_restante(self,t):
        return self.ModelPointRow.duree_restante - t

    @functools.cache
    def taux_nominal(self):
        return self.ModelPointRow.taux_nominal

    @functools.cache
    def taux_mensuel(self):
        return self.ModelPointRow.taux_mensuel
    
    @functools.cache
    def mensualite(self):
        return self.ModelPointRow.mensualite

    @functools.cache
    def produit(self):
        return self.ModelPointRow.produit

    @functools.cache
    def distribution_channel(self):
        return self.ModelPointRow.distribution_channel

    @functools.cache
    def nb_contrats(self):
        return self.ModelPointRow.nb_contrats
    
    @functools.cache
    def age_entree_sinistre(self, t):
        if self.etat()=='v':
            return self.age_actuel(t)
        else : return math.floor(self.age_actuel(t) - self.duree_sinistre(t)/12)

    @functools.cache
    def duree_sinistre(self,t):
        if self.etat()=='v':
            return 0
        return self.ModelPointRow.duree_sinistre + t
    
    ### Hypoth??se demographique
    @functools.cache
    def prob_passage_inc_inv(self, age_entree, duration_incap):
        """retourne la proba de passage de l'??tat inc vers inv

        Args:
            age_entree (int): age entr??e en incap
            duration_incap (int): anciennet?? en incap

        Returns:
            decimal: proba de passer de inc vers inv
        """
        if age_entree>ADECashFLowModel.PassageInval.max_age_pass_inval() or duration_incap>ADECashFLowModel.PassageInval.max_duration_pass_inval():
            return 0
        else:
            return ADECashFLowModel.PassageInval.nombre_passage_inval(age_entree, duration_incap)/ADECashFLowModel.MaintienIncap.nombre_maintien_incap(age_entree, duration_incap)
    
    ### Projection des effectifs
    @functools.cache
    def constitution_matrix_transitions(self, age_entree_etat, duration_etat_at_t, t):
        """Calcul de la matrice des probabilit?? de transitions pour les 6 ??tats : 
        [VALIDE, DC, CHOMAGE, INCAPACITE, INVALIDITE, LAPSE]    x   [VALIDE, DC, CHOMAGE, INCAPACITE, INVALIDITE, LAPSE]
        les ??tats absorbants ont une probabilit?? totale de changement d'??tat = 0 (au lieu de 1) : somme(i,j): j -> 1 ?? n, i constante = 0.
        Ce qui permet de calculer uniquement ?? l'instant t l'effectif des autres ??tats qui ont transit?? dans un ??tat absorbant et non le cumul
        depuis le d??but de la projection
        
        Args:
            age_entree_etat (int): age d'entr??e dans l'??tat en ann??es
            duration_etat_at_t (int): duration dans l'??tat ?? l'instant t en mois
            t (int): instant t de projection

        Returns:
            Matrice: Matrice carr??e des proba de transitions avec l'??l??ment [i,j] est la proba de passer de l'??tat i ?? l'??tat j
        """
        # initialisation de la matrice des transtions ?? zeros
        sex = self.sexe()
        age_actuel = self.age_actuel(t)
        produit = self.produit()
        anc_contrat_mois = self.anciennete_contrat_mois(t)
        mat = np.zeros((6, 6))
        # From valide to ...
        mat[0,1] = ADECashFLowModel.Mortalite.prob_dc(sex, age_actuel)/12
        mat[0,2] = ADECashFLowModel.Incidence.prob_entree_chomage(age_actuel)/12
        mat[0,3] = ADECashFLowModel.Incidence.prob_entree_incap(age_actuel)/12
        mat[0,4] = ADECashFLowModel.Incidence.prob_entree_inval(age_actuel)/12
        mat[0,5] = ADECashFLowModel.Lapse.prob_rachat(produit , anc_contrat_mois)
        mat[0,0] = 1 - np.sum(mat[0,:])
        # From DC to ...
        mat[1,1] = 0
        # From Chomage to ...
        mat[2,1] = ADECashFLowModel.Mortalite.prob_dc(sex, age_actuel)/12
        mat[2,2] = ADECashFLowModel.MaintienCh.prob_passage_ch_ch(age_entree_etat, duration_etat_at_t)
        mat[2,0] = 1 - np.sum(mat[2,:])
        # From incap to ...
        mat[3,1] = ADECashFLowModel.Mortalite.prob_dc(sex, age_actuel)/12
        mat[3,3] = ADECashFLowModel.MaintienIncap.prob_passage_inc_inc(age_entree_etat, duration_etat_at_t)
        mat[3,4] = self.prob_passage_inc_inv(age_entree_etat, duration_etat_at_t)
        mat[3,0] = 1 - np.sum(mat[3,:])
        # From invalidite to ...
        mat[4,4] = 0
        # From Lapse to ...
        mat[5,5] = 0
        return mat
    
    #@functools.cache
    def projection_initiale_state_at_t(self, init_state_at_t0, actual_state_at_t_n_1, mat_transitions):
        """retourne les effectifs projet?? poru chaque ??tat ?? la maille duration

        Args:
            init_state_at_t0 (vecteur(6)): vecteur des effectifs ?? temps t=0. Ex : [0, 12, 0, 0, 0, 0]
            actual_state_at_t_n_1 (vecteur(6)): vecteur des effectifs ?? temps t = t-1
            mat_transitions (matrix(6,6)): matrice de transition entre ??tats

        Returns:
            _type_: _description_
        """
        res = np.zeros((2, mat_transitions.shape[1]))
        tra = np.dot(actual_state_at_t_n_1, mat_transitions)
        res[0,] = (init_state_at_t0>0)*tra
        res[1,] = (init_state_at_t0==0)*tra
        return res

    @functools.cache
    def get_next_state(self, t):
        """Projection des effectifs en temps t.

        Args:
            t (int): temps t de projection

        Returns:
            matrice: repartitions des effectis par anciennet??e
        """
        if t == 0:
            return self.vecteur_des_effectifs_at_t(0).reshape(1,6)
        else:
            mat_transitions = self.constitution_matrix_transitions(self.age_entree_sinistre(t), self.duree_sinistre(t), t)
            init_state_projection = self.projection_initiale_state_at_t(self.vecteur_des_effectifs_at_t(0), self.get_next_state(t-1)[0,:], mat_transitions)
            for row_i in range(1, self.get_next_state(t-1).shape[0]):
                mat_transitions = self.constitution_matrix_transitions(self.age_entree_sinistre(t), row_i, t) # dur??e sinistre = row_i pour mod??liser les nouvelles transtions
                project_at_t = np.dot(self.get_next_state(t-1)[row_i,:], mat_transitions)
                init_state_projection = np.vstack([init_state_projection, project_at_t.reshape(1, mat_transitions.shape[0])])
            init_state_projection[0,0] = np.sum(init_state_projection[:,0])
            init_state_projection[1:,0] = 0
        return init_state_projection
    
    @functools.cache
    def vecteur_des_effectifs_at_t(self, t):
        """Vecteur de effectifs totaux par ??tat temps t:
            6 ??tats suivants : VALIDE, DC, CHOMAGE, INCAPACITE, INVALIDITE, LAPSE

        Returns:
            Vecteur(6) : Effectifs totaux par ??tat sour la forme d'un vecteur [Nombre de VALIDE, Nombre de DC, Nombre de CHOMAGE, Nombre de INCAPACITE, Nombre de INVALIDITE, Nombre de LAPSE]
        """
        if t == 0:
            if self.etat()=='v':
                return self.nb_contrats() * np.array([1, 0, 0, 0, 0, 0])
            if self.etat()=='ch':
                return self.nb_contrats() * np.array([1, 0, 0, 0, 0, 0])
            if self.etat()=='inc':
                return self.nb_contrats() * np.array([1, 0, 0, 0, 0, 0])
        else:
            return  np.sum(self.get_next_state(t), axis=0)

    ### Fonctions utiles des calculs de provisions oslr et prc
    @functools.cache
    def pmxcho(self, age_entre, dure_ecoulee, D1, D2, taux_actu):
        """OSLR  chomage

        Args:
            age_entre (int): Age ?? la survenance du sinistre
            dure_ecoulee (int): Dur??e ??coul??e en incapacit?? en mois
            D1 (int): D??but du paiement dans D1 mois (si pas de franchise ?? 0), nombre de mosi de carence
            D2 (int): Fin du paiement de l'incapacit?? dans D2 mois (cas 1: D2+dureecoulee est inf??rieure ?? 35 mois)
            taux_actu (decimal): Taux technique d'actualisation
        """
        som1 = 0
        som2 = 0
        l = ADECashFLowModel.MaintienCh.nombre_maintien_chomage(age_entre, dure_ecoulee)
        for i in range(D1,D2):
            som1 = som1 + pow((1 + taux_actu),(-(i / 12))) * ADECashFLowModel.MaintienCh.nombre_maintien_chomage(age_entre, dure_ecoulee + i)
            som2 = som2 + pow((1 + taux_actu),(-((i + 1) / 12))) * ADECashFLowModel.MaintienCh.nombre_maintien_chomage(age_entre, dure_ecoulee + i + 1)
        return((som1 + som2) / (2 * l))

    @functools.cache
    def pmxinc(self, agentree, durecoulee, D1, D2, taux_actu):
        """OSLR Incapacite

        Args:
            agentree (int): Age ?? la survenance du sinistre
            durecoulee (int): Dur??e ??coul??e en incapacit?? en mois
            D1 (int): D??but du paiement dans D1 mois (si pas de franchise ?? 0), nombre de mosi de carence
            D2 (int): Fin du paiement de l'incapacit?? dans D2 mois (cas 1: D2+dureecoulee est inf??rieure ?? 35 mois)
            taux_actu (decimal): Taux technique d'actualisation

        Returns:
            _type_: _description_
        """
        som1 = som2 = 0
        l = ADECashFLowModel.MaintienIncap.nombre_maintien_incap(agentree, durecoulee)
        for i in range(D1,D2):
            som1 = som1 + pow((1 + taux_actu),(-(i / 12))) * ADECashFLowModel.MaintienIncap.nombre_maintien_incap(agentree, durecoulee + i)
            som2 = som2 + pow((1 + taux_actu),(-((i + 1) / 12))) * ADECashFLowModel.MaintienIncap.nombre_maintien_incap(agentree, durecoulee + i + 1)
        return ((som1 + som2) / (2 * l))

    @functools.cache
    def pmxpot2(self, agentree, durecoulee, D1, D2, taux, crd):
        """PRC ITT 

        Args:
            agentree (int): Age ?? la survenance du sinistre
            durecoulee (int): Dur??e ??coul??e en incapacit?? en mois
            D1 (int): D??but du paiement dans D1 mois (si pas de franchise ?? 0), nombre de mosi de carence
            D2 (int): Fin du paiement de l'incapacit?? dans D2 mois (cas 1: D2+dureecoulee est inf??rieure ?? 35 mois)
            taux (decimal): Taux d'int??r??t technique
            crd (decimal): Credit restant d??

        Returns:
            decimal: PRC ITT
        """
        som1 = som2 = 0
        l = ADECashFLowModel.MaintienIncap.nombre_maintien_incap(agentree, durecoulee)
        prov = crd
        for i in range(D1,D2):
            som1 = som1 + pow((1 + taux) , -(i / 12)) * ADECashFLowModel.PassageInval.nombre_passage_inval(agentree, durecoulee + i) * prov
            som2 = som2 + pow((1 + taux) , -((i + 1) / 12)) * ADECashFLowModel.PassageInval.nombre_passage_inval(agentree, durecoulee + i + 1) * prov
        return ((som1 + som2) / (2 * l))
    
    # def prc_itt2_v2(self, agentree, D2, taux, crd, mensualite):
    #     """_summary_

    #     Args:
    #         agentree (int): _description_
    #         D2 (int): _description_
    #         taux (decimal): _description_
    #         crd (decimal): _description_
    #         mensualite (decimal): _description_
    #     """
    #     som = 0
    #     for i in range(0,D2+1):
    #         som = som + self.Incidence.prob_entree_incap(round(agentree + i / 12)) * (self.pmxinc(round(agentree + i / 12), 0, 0, D2, taux) * mensualite + self.pmxpot2(round(agentree + i / 12), 0, 0, D2, taux, crd)) * 0.0005 * ((1 + taux) ^ (-(i / 12)))
    #         #lapse(lapse.matrix, caisse, contrat, floor((duration+i)/12)) )
    #     return(som)
    
    @functools.cache
    def prc_inc_clot(self, t, tech_int_rate=0):
        if t> self.duree_restante(0) or t<0:
            return 0
        else:
            eng_assureur = 0
            eng_assure = 0
            duree_restante = self.duree_restante(t)
            age_actuel = self.age_actuel(t)
            sexe = self.sexe()
            produit = self.produit()
            anc_contrat_mois = self.anciennete_contrat_mois(t)
            for i in range(0, duree_restante+1):
                eng_assureur = eng_assureur + (self.Mortalite.lx(sexe, round(age_actuel+i)) / self.Mortalite.lx(sexe, round(age_actuel))) \
                * self.Incidence.prob_entree_incap(round(age_actuel)) / 12 * (self.mensualite() * self.pmxinc(round(age_actuel+i), 0, 0, (35-anc_contrat_mois+i), 0))
            
            for i in range(0, duree_restante+1):
                eng_assure = eng_assure + (self.Mortalite.lx(sexe, round(age_actuel+i)) / self.Mortalite.lx(sexe, round(age_actuel))) * ADECashFLowModel.ReferentielProduit.get_tx_prime_inc(produit) * self.ci() * pow((1+tech_int_rate), (-i))
            return max(eng_assureur-eng_assure, 0) * self.couv_inc()
    
    @functools.cache
    def prc_inc_ouv(self, t, tech_int_rate=0):
        if t==0:
            return self.prc_inc_clot(0, tech_int_rate)
        else : return self.prc_inc_clot(t-1, tech_int_rate)
    
    @functools.cache
    def prc_dc_clot(self, t, tech_int_rate=0):
        """provision pour risk croissant du risque DC

        Args:
            t (int): time t of projection
            tech_int_rate (int, optional): technical interest rate for technical cash flow discount factor. Defaults to 0.

        Returns:
            decimal: provision pour risk croissant
        """
        if t> self.duree_restante(0) or t<0:
            return 0
        else:
            eng_assureur = 0
            eng_assure = 0
            duree_restante = self.duree_restante(t)
            age_actuel = self.age_actuel(t)
            sexe = self.sexe()
            produit = self.produit()
            for i in range(0, duree_restante+1):
                eng_assureur = eng_assureur + (self.Mortalite.lx(sexe, round(age_actuel+i)) / self.Mortalite.lx(sexe, round(age_actuel))) * self.Mortalite.prob_dc(sexe, round(age_actuel+i)) / 12 * self.crd(t+i) * pow((1+tech_int_rate), (-i-0.5))
            
            for i in range(0, duree_restante+1):
                eng_assure = eng_assure + (self.Mortalite.lx(sexe, round(age_actuel+i)) / self.Mortalite.lx(sexe, round(age_actuel))) * ADECashFLowModel.ReferentielProduit.get_tx_prime_dc(produit) * self.ci() * pow((1+tech_int_rate), (-i))
            return max(eng_assureur-eng_assure, 0) * self.couv_dc()
    
    @functools.cache
    def prc_dc_ouv(self, t, tech_int_rate=0):
        if t==0:
            return self.prc_dc_clot(0, tech_int_rate)
        else : return self.prc_dc_clot(t-1, tech_int_rate)
    
    @functools.cache
    def amortisation_schedule(self, amount, annualinterestrate, paymentsperyear, years):
        """
            Tableau d'amortissement du pr??t
        Args:
            amount (decimal): Montant initial du pr??t
            annualinterestrate (decimal): taux d'interet annuel
            paymentsperyear (decimal): Nombre de paiment dans l'ann??e
            years (int): nombre d'ann??es du pr??t

        Returns:
            Pandas DataFrame: Tableau d'amortissement du pr??t
        """
        df = pd.DataFrame({'PrincipalPaid' :[npf.ppmt(annualinterestrate/paymentsperyear, i+1, paymentsperyear*years, amount) for i in range(paymentsperyear*years)],
                           'InterestPaid' :[npf.ipmt(annualinterestrate/paymentsperyear, i+1, paymentsperyear*years, amount) for i in range(paymentsperyear*years)]})
        df['Instalment'] = df.PrincipalPaid + df.InterestPaid
        df['CumulativePrincipal'] = np.cumsum(df.PrincipalPaid)
        df['Principal'] = amount
        df['Balance'] = df['Principal'] + df['CumulativePrincipal']
        df['Mois'] = np.arange(1, df.shape[0]+1, 1)
        return (df)
    
    @functools.cache
    def loan_balance_at_n(self, amount, annualinterestrate, paymentsperyear, years, n):
        """montant/Capital restat d?? juste apr??s le n-ieme remboursement p??riodique

        Args:
            amount (decimal): Montant initial du pr??t
            annualinterestrate (flott): taux d'inter??t annuel du pret
            paymentsperyear (int): nombre de mensualit??s dans l'ann??e
            years (int): nombre d'ann??es du pr??t
            n (int): N-ieme paiement effectu??

        Returns:
            decimal: Capital restant d?? juste apr??s le (n)-ieme paiement
        """
        principalpaid = [npf.ppmt(annualinterestrate/paymentsperyear, i+1, paymentsperyear*years, amount) for i in range(n)]
        return amount+np.sum(principalpaid)
    
    @functools.cache
    def couv_inv(self):
        """couverture invalidit?? 

        Returns:
            bool: 0 si non couv inv et 1 si couv inv
        """
        if ADECashFLowModel.ReferentielProduit.get_tx_prime_inc(self.produit())==0:
            return 0
        else:
            return 1
    
    @functools.cache
    def couv_inc(self):
        """couverture incapacit?? 

        Returns:
            bool: 0 si non couv inc et 1 si couv inc
        """
        if ADECashFLowModel.ReferentielProduit.get_tx_prime_inc(self.produit())==0:
            return 0
        else:
            return 1
    
    @functools.cache
    def couv_ch(self):
        """couverture chomage 

        Returns:
            bool: 0 si non couv ch et 1 si couv ch
        """
        if ADECashFLowModel.ReferentielProduit.get_tx_prime_chomage(self.produit())==0:
            return 0
        else:
            return 1
    
    @functools.cache
    def couv_dc(self):
        """couverture DC 

        Returns:
            bool: 0 si non couv dc et 1 si couv dc
        """
        if ADECashFLowModel.ReferentielProduit.get_tx_prime_dc(self.produit())==0:
            return 0
        else:
            return 1
        
    def projection_des_effectif_du_mp(self):
        """Projection du model point ?? la fin du contrat et ou des garanties

        Returns:
            DataFrame: variables du mp projet??es dans le temps
        """
        ultim = self.duree_restante(0)
        res = {'time':[t for t in range(ultim)],
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
                'produit':[self.produit() for t in range(ultim)],
                'distribution_channel':[self.distribution_channel() for t in range(ultim)],
                'nb_contrats':[np.sum(self.vecteur_des_effectifs_at_t(t)) for t in range(ultim)],
                'duree_sinistre':[self.duree_sinistre(t) for t in range(ultim)],
                'v':[self.vecteur_des_effectifs_at_t(t)[0] for t in range(ultim)],
                'dc':[self.vecteur_des_effectifs_at_t(t)[1] for t in range(ultim)],
                'ch':[self.vecteur_des_effectifs_at_t(t)[2] for t in range(ultim)],
                'inc':[self.vecteur_des_effectifs_at_t(t)[3] for t in range(ultim)],
                'inv':[self.vecteur_des_effectifs_at_t(t)[4] for t in range(ultim)],
                'lps':[self.vecteur_des_effectifs_at_t(t)[5] for t in range(ultim)],
                'crd':[self.crd(t) for t in range(ultim)],
                'couv_dc':[self.couv_dc() for t in range(ultim)],
                'couv_inv':[self.couv_inv() for t in range(ultim)],
                'couv_inc':[self.couv_inc() for t in range(ultim)],
                'couv_ch':[self.couv_ch() for t in range(ultim)]
                }
        return pd.DataFrame.from_dict(res)
    
    def calcul_des_flux_du_mp(self):
        """"Retourne Projection du model point ?? la fin du contrat et ou des garanties avec les flux financiers

        Returns:
            DataFrame: Projection du model point ?? la fin du contrat et ou des garanties avec les flux financiers
        """
        #df = self.projection_des_effectif_du_mp()
        df = pd.DataFrame.from_dict(self.projection_des_effectif_du_mp_3())
        produit = self.produit()
        print("1")
        # flux sinistres et frais gestion sinistres
        df["sinistre_dc"] = df.apply(lambda x: x.dc * x.crd * x.couv_dc if (x.anciennete_contrat_annee<=30) or (x.age_actuel<=75) else 0, axis=1)
        df["sinistre_inv"] = df.apply(lambda x: x.inv * x.crd * x.couv_inv if (x.anciennete_contrat_annee<=30) or (x.age_actuel<=60) else 0, axis=1)
        df["sinistre_ch"] = df.apply(lambda x: x.ch * x.mensualite * x.couv_ch if (x.anciennete_contrat_annee<=30) or (x.age_actuel<=65) else 0, axis=1)
        df["sinistre_inc"] = df.apply(lambda x: x.inc * x.mensualite * x.couv_inc if (x.anciennete_contrat_annee<=30) or (x.age_actuel<=65)  else 0, axis=1)
        df["total_sinistre"] = df["sinistre_dc"]+df["sinistre_inv"]+df["sinistre_ch"]+df["sinistre_inc"]
        df["frais_gest_sin"] = df["total_sinistre"] * ADECashFLowModel.ReferentielProduit.get_tx_frais_gest_sin(produit)
        # flux primes
        print("2")
        df["primes_valides"] = df.v * (ADECashFLowModel.ReferentielProduit.get_tx_prime_chomage(produit)+ADECashFLowModel.ReferentielProduit.get_tx_prime_dc(produit)+ADECashFLowModel.ReferentielProduit.get_tx_prime_inc(produit)) * df.ci
        df["primes_inc"] = df.inc * (ADECashFLowModel.ReferentielProduit.get_tx_prime_chomage(produit)+ADECashFLowModel.ReferentielProduit.get_tx_prime_dc(produit)+ADECashFLowModel.ReferentielProduit.get_tx_prime_inc(produit)) * df.ci
        df["primes_ch"] = df.ch * (ADECashFLowModel.ReferentielProduit.get_tx_prime_chomage(produit)+ADECashFLowModel.ReferentielProduit.get_tx_prime_dc(produit)+ADECashFLowModel.ReferentielProduit.get_tx_prime_inc(produit)) * df.ci
        df["total_primes"] = df["primes_valides"]+df["primes_inc"]+df["primes_ch"]
        #flux frais sur primes
        print("3")
        df["frais_administrations"] = ADECashFLowModel.ReferentielProduit.get_tx_frais_admin(produit) * df["total_primes"]
        df["frais_acquisitions"] = ADECashFLowModel.ReferentielProduit.get_tx_frais_acq(produit) * df["total_primes"]
        df["frais_commissions"] = ADECashFLowModel.ReferentielProduit.get_tx_comm(produit) * df["total_primes"]
        df["total_frais_primes"] = df["frais_commissions"] + df["frais_acquisitions"] + df["frais_administrations"]
        return df
    
    @functools.cache
    def sinistre_dc(self, t):
        """Montant du sinistre pour DC

        Args:
            t (int): temps de projection

        Returns:
            decimal: Montant du sinistre pour DC ?? t
        """
        ultimate = self.duree_restante(0)
        if (self.anciennete_contrat_annee(t)<=30 or self.age_actuel(t)<=75) and t<=ultimate:
            return self.vecteur_des_effectifs_at_t(t)[1] * self.couv_dc() * self.crd(t)
        else: return 0
        
    @functools.cache
    def sinistre_inv(self, t):
        """Montant du sinistre pour inv

        Args:
            t (int): temps de projection

        Returns:
            decimal: Montant du sinistre pour inv ?? t
        """
        ultimate = self.duree_restante(0)
        if (self.anciennete_contrat_annee(t)<=30 or self.age_actuel(t)<=60) and t<=ultimate:
            return self.vecteur_des_effectifs_at_t(t)[4] * self.couv_inv() * self.crd(t)
        else: return 0
        
    @functools.cache
    def sinistre_ch(self, t):
        """Montant du sinistre pour ch

        Args:
            t (int): temps de projection

        Returns:
            decimal: Montant du sinistre pour ch ?? t
        """
        ultimate = self.duree_restante(0)
        if (self.anciennete_contrat_annee(t)<=30 or self.age_actuel(t)<=65) and t<=ultimate:
            return self.vecteur_des_effectifs_at_t(t)[2] * self.couv_ch() * self.mensualite()
        else: return 0
        
    @functools.cache
    def sinistre_inc(self, t):
        """Montant du sinistre pour inc

        Args:
            t (int): temps de projection

        Returns:
            decimal: Montant du sinistre pour inc ?? t
        """
        ultimate = self.duree_restante(0)
        if (self.anciennete_contrat_annee(t)<=30 or self.age_actuel(t)<=65) and t<=ultimate:
            return self.vecteur_des_effectifs_at_t(t)[3] * self.couv_inc() * self.mensualite()
        else: return 0
        
    @functools.cache
    def total_sinistre(self, t):
        if t<=self.duree_restante(0):
            return self.sinistre_dc(t) + self.sinistre_inv(t) + self.sinistre_inc(t) + self.sinistre_ch(t)
        else: return 0
    
    @functools.cache
    def frais_gest_sin(self, t):
        """retourne le Montant total sinistre dc+inv+inc+ch ?? t

        Args:
            t (int): temps t

        Returns:
            decimal: Montant total sinistre dc+inv+inc+ch ?? t
        """
        if t<=self.duree_restante(0):
            return self.total_sinistre(t) * ADECashFLowModel.ReferentielProduit.get_tx_frais_gest_sin(self.produit())
        else: return 0
    
    @functools.cache
    def prime_valide(self, t):
        """montant des primes vers??es par les valides

        Args:
            t (int): temps t

        Returns:
            decimal: montant des primes vers??es par les valides ?? t
        """
        produit = self.produit()
        if t<=self.duree_restante(0):
            return self.vecteur_des_effectifs_at_t(t)[0] * (ADECashFLowModel.ReferentielProduit.get_tx_prime_chomage(produit)+ADECashFLowModel.ReferentielProduit.get_tx_prime_dc(produit)+ADECashFLowModel.ReferentielProduit.get_tx_prime_inc(produit)) * self.ci() * self.couv_inv()
        else: return 0
    
    @functools.cache
    def prime_inc(self, t):
        """montant des primes vers??es par les effectifs au incapacit??s

        Args:
            t (int): temps t

        Returns:
            decimal: montant des primes vers??es par les effectifs en incapacit??s ?? t
        """
        produit = self.produit()
        if t<=self.duree_restante(0):
            return self.vecteur_des_effectifs_at_t(t)[3] * (ADECashFLowModel.ReferentielProduit.get_tx_prime_chomage(produit)+ADECashFLowModel.ReferentielProduit.get_tx_prime_dc(produit)+ADECashFLowModel.ReferentielProduit.get_tx_prime_inc(produit)) * self.ci() * self.couv_inc()
        else: return 0
        
    @functools.cache
    def prime_ch(self, t):
        """montant des primes vers??es par les effectifs au chomages

        Args:
            t (int): temps t

        Returns:
            decimal: montant des primes vers??es par les effectifs au chomages ?? t
        """
        produit = self.produit()
        if t<=self.duree_restante(0):
            return self.vecteur_des_effectifs_at_t(t)[2] * (ADECashFLowModel.ReferentielProduit.get_tx_prime_chomage(produit)+ADECashFLowModel.ReferentielProduit.get_tx_prime_dc(produit)+ADECashFLowModel.ReferentielProduit.get_tx_prime_inc(produit)) * self.ci() * self.couv_ch()
        else: return 0
    
    @functools.cache
    def total_prime(self, t):
        """montant total des primes vers??es par les effectifs.
        Args:
            t (int): temps t

        Returns:
            decimal: montant total des primes vers??es par les effectifs.
        """
        if t<=self.duree_restante(0):
            return self.prime_valide(t) + self.prime_inc(t) + self.prime_ch(t)
        else: return 0
        
    @functools.cache
    def frais_administrations(self, t):
        """montant  des frais d'administrations ?? t.
        Args:
            t (int): temps t

        Returns:
            decimal: montant des frais d'administrations ?? t.
        """
        if t<=self.duree_restante(0):
            return ADECashFLowModel.ReferentielProduit.get_tx_frais_admin(self.produit()) * self.total_prime(t)
        else: return 0
    
    @functools.cache
    def frais_acquisitions(self, t):
        """montant  des frais d'acquisisiton ?? t.
        Args:
            t (int): temps t

        Returns:
            decimal: montant des frais d'acquisisiton ?? t.
        """
        if t<=self.duree_restante(0):
            return ADECashFLowModel.ReferentielProduit.get_tx_frais_acq(self.produit()) * self.total_prime(t)
        else: return 0
    
    @functools.cache
    def frais_commissions(self, t):
        """montant des frais de commission ?? t.
        Args:
            t (int): temps t

        Returns:
            decimal: montant des frais de commission ?? t.
        """
        if t<=self.duree_restante(0):
            return ADECashFLowModel.ReferentielProduit.get_tx_comm(self.produit()) * self.total_prime(t)
        else: return 0
    
    @functools.cache
    def total_frais_primes(self, t):
        """montant total des frais ?? t.
        Args:
            t (int): temps t

        Returns:
            decimal: montant total des frais ?? t.
        """
        if t<=self.duree_restante(0):
            return self.frais_administrations(t) + self.frais_acquisitions(t) + self.frais_commissions(t)
        else: return 0
    
    @functools.cache
    def pm_inc_clo(self, t):
        """montant des pm oslr inc ?? la cloture de t

        Args:
            t (int): temps t

        Returns:
            decimal: montant des pm oslr inc ?? la cloture de t
        """
        if t<=self.duree_restante(0):
            return self.pmxinc(math.floor(self.age_actuel(t)), self.duree_sinistre(t), 0, (35-self.anciennete_contrat_mois(t)), 0) * self.vecteur_des_effectifs_at_t(t)[3] * self.couv_inc()
        else: return 0
    
    @functools.cache
    def pm_inc_ouv(self, t):
        """montant des pm oslr inc ?? l'ouverture de t

        Args:
            t (int): temps t

        Returns:
            decimal: montant des pm oslr inc ?? l'ouverture de t
        """
        if t==0:
            return self.pm_inc_clo(0)
        else : return self.pm_inc_clo(t-1)
    
    @functools.cache
    def pm_cho_clo(self, t):
        """montant des pm oslr cho ?? la cloture de t

        Args:
            t (int): temps t

        Returns:
            decimal: montant des pm oslr cho ?? la cloture de t
        """
        if t<=self.duree_restante(0):
            return self.pmxcho(math.floor(self.age_actuel(t)), self.duree_sinistre(t), 0, 35-self.anciennete_contrat_mois(t), 0) * self.vecteur_des_effectifs_at_t(t)[2] * self.couv_ch()
        else: return 0
    
    @functools.cache
    def pm_cho_ouv(self, t):
        """montant des pm oslr cho ?? la l'ouverture de t

        Args:
            t (int): temps t

        Returns:
            decimal: montant des pm oslr cho ?? la l'ouverture de t
        """
        if t==0:
            return self.pm_cho_clo(0)
        else : return self.pm_cho_clo(t-1)
    
    # def pm_inc_inv_clo(self, t):
    #     """montant des pm oslr inc-inv ?? la cloture de t

    #     Args:
    #         t (int): temps t

    #     Returns:
    #         decimal: montant des pm oslr inc-inv ?? la cloture de t
    #     """
    #     if t<=self.duree_restante(0):
    #         return self.pmxpot2(math.floor(self.age_actuel(t)), self.duree_sinistre(t), 0, 35-self.anciennete_contrat_mois(t), 0, self.crd(t)) * self.vecteur_des_effectifs_at_t(t)[3] * self.couv_inv()
    #     else: return 0
        
    # def pm_inc_inv_ouv(self, t):
    #     """montant des pm oslr inc-inv ?? l'ouverture de t

    #     Args:
    #         t (int): temps t

    #     Returns:
    #         decimal: montant des pm oslr inc-inv ?? l'ouverture de t
    #     """
    #     if t==0:
    #         return self.pm_inc_inv_clo(0)
    #     else : return self.pm_inc_inv_clo(t-1)
    
    @functools.cache
    def total_provisions_clot(self, t):
        """retourne le total des provisions ?? la cloture

        Args:
            t (int): temps t

        Returns:
            decimal: retourne le total des provisions ?? la cloture
        """
        if t<=self.duree_restante(0):
            return self.pm_inc_clo(t) + self.pm_cho_clo(t) + self.prc_inc_clot(t) + self.prc_dc_clot(t)
        else: return 0
    
    @functools.cache
    def total_provisions_ouv(self, t):
        """reoturne le total des provisions ?? l'ouverture

        Args:
            t (int): temps

        Returns:
            decimal: total des provisions ?? l'ouverture
        """
        if t<=self.duree_restante(0):
            return self.pm_inc_ouv(t) + self.pm_cho_ouv(t) + self.prc_inc_ouv(t) + self.prc_dc_ouv(t)
        else: return 0
    
    @functools.cache
    def production_financiere(self, t):
        """retourne la production financi??re moyenne entre l'ouverture et la cloture

        Args:
            t (int): temps t

        Returns:
            decimal: retourne la production financi??re moyenne entre l'ouverture et la cloture
        """
        if t> self.duree_restante(0):
            return 0
        else:
            return (self.total_provisions_clot(t) + self.total_provisions_ouv(t)) / 2 * ADECashFLowModel.ReferentielProduit.get_tx_production_financiere(self.produit())
    
    @functools.cache
    def resultat_technique(self, t):
        """retourne le resultat technique : primes - frais - sinistres - les augmentations de pm entre l'ouv et la cloture

        Args:
            t (int): temps t

        Returns:
            _type_: _description_
        """
        if t> self.duree_restante(0): 
            return 0
        else:
            return self.total_prime(t) - (self.total_frais_primes(t) + self.frais_gest_sin(t)) - self.total_sinistre(t) \
                                    - (self.pm_inc_clo(t)-self.pm_inc_ouv(t)) \
                                    - (self.pm_cho_clo(t)-self.pm_cho_ouv(t)) \
                                    - (self.prc_dc_clot(t)-self.prc_dc_ouv(t)) \
                                    - (self.prc_inc_clot(t)-self.prc_inc_ouv(t))
    
    @functools.cache               
    def participation_benef_ass(self, t):
        """participation au benef assureur

        Args:
            t (int): temps t

        Returns:
            decimal: participation au benef assureur
        """
        if t> self.duree_restante(0):
            return 0
        else:
            return self.resultat_technique(t) * ADECashFLowModel.ReferentielProduit.get_tx_profit_sharing_assureur(self.produit())
    
    @functools.cache
    def participation_benef_part(self, t):
        """participation au benef partenaire

        Args:
            t (int): temps t

        Returns:
            decimal: participation au benef partenaire 
        """
        if t> self.duree_restante(0):
            return 0
        else:
            return self.resultat_technique(t) * ADECashFLowModel.ReferentielProduit.get_tx_profit_sharing_partenaire(self.produit())
    
    @functools.cache
    def resultat_assureur(self, t):
        if t> self.duree_restante(0):
            return 0
        else:
            return self.resultat_technique(t)+self.participation_benef_ass(t)+self.production_financiere(t)
        
    def full_cash_flow_projection(self):
        """Projection de toutes variables du model point ?? la fin du contrat et ou des garanties

        Returns:
            DataFrame: variables du mp projet??es dans le temps
        """
        ultim = self.duree_restante(0)
        df = {'time':[t for t in range(ultim)],
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
                           'produit':[self.produit() for t in range(ultim)],
                           'distribution_channel':[self.distribution_channel() for t in range(ultim)],
                           'nb_contrats':[np.sum(self.vecteur_des_effectifs_at_t(t)) for t in range(ultim)],
                           'duree_sinistre':[self.duree_sinistre(t) for t in range(ultim)],
                           'v':[self.vecteur_des_effectifs_at_t(t)[0] for t in range(ultim)],
                           'dc':[self.vecteur_des_effectifs_at_t(t)[1] for t in range(ultim)],
                           'ch':[self.vecteur_des_effectifs_at_t(t)[2] for t in range(ultim)],
                           'inc':[self.vecteur_des_effectifs_at_t(t)[3] for t in range(ultim)],
                           'inv':[self.vecteur_des_effectifs_at_t(t)[4] for t in range(ultim)],
                           'lps':[self.vecteur_des_effectifs_at_t(t)[5] for t in range(ultim)],
                           'crd':[self.crd(t) for t in range(ultim)],
                           'couv_dc':[self.couv_dc() for t in range(ultim)],
                           'couv_inv':[self.couv_inv() for t in range(ultim)],
                           'couv_inc':[self.couv_inc() for t in range(ultim)],
                           'couv_ch':[self.couv_ch() for t in range(ultim)],
                           'prime_valide':[self.prime_valide(t) for t in range(ultim)],
                           'prime_inc':[self.prime_inc(t) for t in range(ultim)],
                           'prime_ch':[self.prime_ch(t) for t in range(ultim)],
                           'total_prime':[self.total_prime(t) for t in range(ultim)],
                           'frai_acq':[self.frais_acquisitions(t) for t in range(ultim)],
                           'frais_admin':[self.frais_administrations(t) for t in range(ultim)],
                           'frais_commissions':[self.frais_commissions(t) for t in range(ultim)],
                           'total_frais_primes':[self.total_frais_primes(t) for t in range(ultim)],
                           'sinistre_dc':[self.sinistre_dc(t) for t in range(ultim)],
                           'sinistre_inc':[self.sinistre_inc(t) for t in range(ultim)],
                           'sinistre_ch':[self.sinistre_ch(t) for t in range(ultim)],
                           'sinistre_inv':[self.sinistre_inv(t) for t in range(ultim)],
                           'sinistre_total':[self.total_sinistre(t) for t in range(ultim)],
                           'frais_gest_sin':[self.frais_gest_sin(t) for t in range(ultim)],
                           'pm_inc_clo':[self.pm_inc_clo(t) for t in range(ultim)],
                           'pm_ch_clo':[self.pm_cho_clo(t) for t in range(ultim)],
                           'prc_inc_clo':[self.prc_inc_clot(t) for t in range(ultim)],
                           'prc_dc_clo':[self.prc_dc_clot(t) for t in range(ultim)],
                           'total_pm_clot':[self.total_provisions_clot(t) for t in range(ultim)],
                           'prc_inc_ouv':[self.prc_inc_ouv(t) for t in range(ultim)],
                           'prc_dc_ouv':[self.prc_dc_ouv(t) for t in range(ultim)],
                           'pm_inc_ouv':[self.pm_inc_ouv(t) for t in range(ultim)],
                           'pm_ch_ouv':[self.pm_cho_ouv(t) for t in range(ultim)],
                           'total_pm_ouv':[self.total_provisions_ouv(t) for t in range(ultim)],
                           'production_financiere':[self.production_financiere(t) for t in range(ultim)],
                           'resultat_technique':[self.resultat_technique(t) for t in range(ultim)],
                           'participation_benef_ass':[self.participation_benef_ass(t) for t in range(ultim)],
                           'participation_benef_part':[self.participation_benef_part(t) for t in range(ultim)],
                           'resultat_assureur':[self.resultat_assureur(t) for t in range(ultim)]
                           }
        return pd.DataFrame.from_dict(df)
    
    def pv_prc_inc_ouv(self):
        """present value of PRC incapacite ?? l'ouverture

        Returns:
            decimal: present value of PRC incapacite ?? l'ouverture
        """
        ultim = self.duree_restante(0)
        return sum([self.prc_inc_ouv(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_prc_dc_ouv(self):
        """present value of prc dc ?? l'ouverture

        Returns:
            decimal: present value of prc dc ?? l'ouverture
        """
        ultim = self.duree_restante(0)
        return sum([self.prc_dc_ouv(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_pm_inc_ouv(self):
        """present value de la pm incapacite ?? l'ouverture

        Returns:
            decimal: present value de la pm incapacite ?? l'ouverture
        """
        ultim = self.duree_restante(0)
        return sum([self.pm_inc_ouv(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_pm_ch_ouv(self):
        """present value de la pm chomage ?? l'ouverture

        Returns:
            decimal: present value de la pm chomage ?? l'ouverture
        """
        ultim = self.duree_restante(0)
        return sum([self.pm_cho_ouv(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_total_pm_ouv(self):
        """present value de la pm total ?? l'ouverture

        Returns:
            decimal: present value de la pm total ?? l'ouverture
        """
        ultim = self.duree_restante(0)
        return sum([self.total_provisions_ouv(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_pm_inc_clo(self):
        """present value de la pm incapcit?? ?? la cloture

        Returns:
            decimal: present value de la pm incapcit?? ?? la cloture
        """
        ultim = self.duree_restante(0)
        return sum([self.pm_inc_clo(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_pm_ch_clo(self):
        """present value de la pm chomage ?? la cloture

        Returns:
            decimal: present value de la pm chomage ?? la cloture
        """
        ultim = self.duree_restante(0)
        return sum([self.pm_cho_clo(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_prc_inc_clo(self):
        """present value de la prc incapacit?? ?? la cloture

        Returns:
            decimal: present value de la prc incapacit?? ?? la cloture
        """
        ultim = self.duree_restante(0)
        return sum([self.prc_inc_ouv(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_prc_dc_clo(self):
        """present value de la prc dc ?? la cloture

        Returns:
            decimal: present value de la prc dc ?? la cloture
        """
        ultim = self.duree_restante(0)
        return sum([self.prc_dc_clo(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_total_pm_clot(self):
        """present value de la pm total ?? la cloture

        Returns:
            decimal: present value de la pm total ?? la cloture
        """
        ultim = self.duree_restante(0)
        return sum([self.total_provisions_clot(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_prime_valide(self):
        """present value de la prime valide ?? la cloture

        Returns:
            decimal: present value de la prime valide ?? la cloture
        """
        ultim = self.duree_restante(0)
        return sum([self.prime_valide(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_prime_inc(self):
        ultim = self.duree_restante(0)
        return sum([self.prime_inc(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_prime_ch(self):
        """present valude la prime vers??e par les indiv en ??tat de chomage

        Returns:
            decimal: present valude la prime vers??e par les indiv en ??tat de chomage
        """
        ultim = self.duree_restante(0)
        return sum([self.prime_ch(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_total_prime(self):
        """present value du total des primes

        Returns:
            decimal: present value du total des primes
        """
        ultim = self.duree_restante(0)
        return sum([self.total_prime(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_frais_acq(self):
        """present value des frais d'acquisition

        Returns:
            decimal: present value des frais d'acquisition
        """
        ultim = self.duree_restante(0)
        return sum([self.frais_acquisitions(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_frais_admin(self):
        """present value des frais administrations

        Returns:
            decimal: present value des frais administrations
        """
        ultim = self.duree_restante(0)
        return sum([self.frais_administrations(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_frais_commissions(self):
        """present value des frais de commissions

        Returns:
            decimal: present value des frais de commissions
        """
        ultim = self.duree_restante(0)
        return sum([self.frais_commissions(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_total_frais_primes(self):
        """present value des frais totaux sur primes

        Returns:
            decimal: present value des frais totaux sur primes
        """
        ultim = self.duree_restante(0)
        return sum([self.total_frais_primes(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_sinistre_dc(self):
        """present value des sinistres dc

        Returns:
            decimal: present value des sinistres dc
        """
        ultim = self.duree_restante(0)
        return sum([self.sinistre_dc(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_sinistre_inc(self):
        """present valude des sinistres sur incapacit??s

        Returns:
            decimal: present valude des sinistres sur incapacit??s
        """
        ultim = self.duree_restante(0)
        return sum([self.sinistre_inc(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_sinistre_ch(self):
        """present valude des sinistres sur chomages

        Returns:
            decimal: present valude des sinistres sur chomages
        """
        ultim = self.duree_restante(0)
        return sum([self.sinistre_ch(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_sinistre_inv(self):
        """present value des sinistres sur invalidit??s

        Returns:
            decimal: present value des sinistres sur invalidit??s
        """
        ultim = self.duree_restante(0)
        return sum([self.sinistre_inv(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_sinistre_total(self):
        """present value des sinistres totaux

        Returns:
            decimal: present value des sinistres totaux
        """
        ultim = self.duree_restante(0)
        return sum([self.total_sinistre(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_frais_gest_sin(self):
        """present value des frais gestion de sinistres

        Returns:
            decimal: present value des frais gestion de sinistres
        """
        ultim = self.duree_restante(0)
        return sum([self.frais_gest_sin(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_production_financiere(self):
        """present valude de la production financiere

        Returns:
            decimal: present valude de la production financiere
        """
        ultim = self.duree_restante(0)
        return sum([self.production_financiere(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_resultat_technique(self):
        """present value du resultat technique

        Returns:
            decimal: present value du resultat technique
        """
        ultim = self.duree_restante(0)
        return sum([self.resultat_technique(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_participation_benef_ass(self):
        """present value de la participation au benef assur??

        Returns:
            decimal: present value de la participation au benef assur??
        """
        ultim = self.duree_restante(0)
        return sum([self.participation_benef_ass(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_participation_benef_part(self):
        """present value de la participation au benef part

        Returns:
            decimal: present value de la participation au benef part
        """
        ultim = self.duree_restante(0)
        return sum([self.participation_benef_part(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
    
    def pv_resultat_assureur(self):
        """present value du resutat assureur

        Returns:
            decimal: present value du resutat assureur
        """
        ultim = self.duree_restante(0)
        return sum([self.resultat_assureur(t)*ADECashFLowModel.ForwardRate.fact_actu(t) for t in range(ultim)])
        
    def present_value_future_cash_flow(self):
        """present value of future cash flows

        Returns:
            Dataframe: present value of future cash flows
        """
        df = {'time':0,
              'mp_id':self.mp_id(),
              'last_t':self.duree_restante(0),
              'pv_prime_valide':self.pv_prime_valide(),
              'pv_prime_inc':self.pv_prime_inc(),
              'pv_prime_ch':self.pv_prime_ch(),
              'pv_total_prime':self.pv_total_prime(),
              'pv_frais_acq':self.pv_frais_acq(),
              'pv_frais_admin':self.pv_frais_admin(),
              'pv_frais_commissions':self.pv_frais_commissions(),
              'pv_total_frais_primes':self.pv_total_frais_primes(),
              'pv_sinistre_dc':self.pv_sinistre_dc(),
              'pv_sinistre_inc':self.pv_sinistre_inc(),
              'pv_sinistre_ch':self.pv_sinistre_ch(),
              'pv_sinistre_inv':self.pv_sinistre_inv(),
              'pv_sinistre_total':self.pv_sinistre_total(),
              'pv_total_pm_clot':self.pv_total_pm_clot(),
              'pv_total_pm_ouv':self.pv_total_pm_ouv(),
              'pv_frais_gest_sin':self.pv_frais_gest_sin(),
              'pv_production_financiere':self.pv_production_financiere(),
              'pv_resultat_technique':self.pv_resultat_technique(),
              'pv_participation_benef_ass':self.pv_participation_benef_ass(),
              'pv_participation_benef_part':self.pv_participation_benef_part(),
              'pv_resultat_assureur':self.pv_resultat_assureur()}
        return pd.DataFrame.from_dict([df])


if __name__=="__main__":
    #Dictionnay containing input files paths
    dataconfig ={'IncidenceData' : 'C://Users//work//OneDrive//modele_emprunteur//CSV//INCIDENCE.csv',
            'LapseData' : 'C://Users//work//OneDrive//modele_emprunteur//CSV//LAPSE.csv',
            'MaintienChData' : 'C://Users//work//OneDrive//modele_emprunteur//CSV//MAINTIEN_CH.csv',
            'MaintienIncapData' : 'C://Users//work//OneDrive//modele_emprunteur//CSV//MAINTIEN_INCAP.csv',
            'MortaliteTHData' : 'C://Users//work//OneDrive//modele_emprunteur//CSV//MORTALITE_TF0002.csv',
            'MortaliteTFData' : 'C://Users//work//OneDrive//modele_emprunteur//CSV//MORTALITE_TH0002.csv',
            'PassageInvalData' : 'C://Users//work//OneDrive//modele_emprunteur//CSV//PASSAGE_INVAL.csv',
            'referentielProduit' : 'C://Users//work//OneDrive//modele_emprunteur//CSV//TBL_PROD.csv',
            'TxForward' : 'C://Users//work//OneDrive//modele_emprunteur//CSV//TX_FORWARD.csv',
            'PathOutput' : 'C://Users//work//OneDrive//modele_emprunteur//CSV//SORTIES_MODELE//'}
    # Set the model configuration
    ADECashFLowModel.config(dataconfig)
    # Read the Model point file into dataframe
    ModelPoint = pd.read_csv('C://Users//work//OneDrive//modele_emprunteur//CSV//MODEL_POINT_2.csv', sep=";")
    # Initialisaiton of the result dataframe
    run_result = pd.DataFrame()
    # projection = ADECashFLowModel(ModelPoint.loc[2,:])
    # print(projection.present_value_future_cash_flow())
    for i in range(0, ModelPoint.shape[0]):
        print(i)
        projection = ADECashFLowModel(ModelPoint.loc[i,:])
        # run_result = run_result.append(projection.full_cash_flow_projection())
        run_result = pd.concat([run_result, projection.present_value_future_cash_flow()], axis=0, join='outer')
        print(i)
    date_time = datetime.now().strftime("%d%m%Y_%H%M%S")
    path_string = ADECashFLowModel.ModelConfig['PathOutput']+'run_ae_'+date_time+'.csv'
    run_result.to_csv(path_string, sep=',', decimal='.', header=True, index=False)