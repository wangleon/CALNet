import math
import numpy as np

class LS(object):
    '''Class for Lomb-Scargle periodogram
    '''

    def __init__(self, t_lst, y_lst):
        '''
        Args:
            t_lst (:class:`numpy.array`): List of time.
            y_lst (:class:`numpy.array`): List of *y* values.
        '''
        self.t_lst = t_lst
        self.y_lst = y_lst
        self.N = len(self.t_lst)

    def get_power(self, period):
        '''Get power spectrum.
        Args:
            period (:class:`numpy.array`): Period list to calculate the power
                spectrum.
        Returns:
            :class:`numpy.array`: Powers at given periods.
        '''
        omega_lst = 2*math.pi/period

        # variable alias
        t = self.t_lst
        y = self.y_lst - self.y_lst.mean(dtype=np.float64)
        YY = (y**2).sum(dtype=np.float64)

        power_lst = []

        for omega in omega_lst:

            sum_sin2wt = (np.sin(2*omega*t)).sum(dtype=np.float64)
            sum_cos2wt = (np.cos(2*omega*t)).sum(dtype=np.float64)
            tau = np.arctan2(sum_sin2wt, sum_cos2wt)/2/omega

            cost = np.cos(omega*(t - tau))
            sint = np.sin(omega*(t - tau))
            YC = (y*cost).sum(dtype=np.float64)
            YS = (y*sint).sum(dtype=np.float64)
            CC = (cost**2).sum(dtype=np.float64)
            SS = (sint**2).sum(dtype=np.float64)
            power = (YC**2/CC + YS**2/SS)/YY
            power_lst.append(power)

        power_lst = np.array(power_lst)

        return power_lst

class GLS(object):
    '''
    Class for generalized Lomb-Scargle periodogram (Zechmeister & Kürster,
    2009).
    see Zechmeister & Kürster, 2009, A&A, 496, 577
    '''
    def __init__(self, t_lst, y_lst, yerr_lst=None, norm='HorneBaliunas'):
        '''
        Args:
            t_lst (:class:`numpy.array`): List of time.
            y_lst (:class:`numpy.array`): List of *y* values.
            yerr_lst (:class:`numpy.array`): List of uncertainties on *y*.
            norm (string): Method of normalization.
        '''
        self.t_lst    = t_lst
        self.y_lst    = y_lst
        if yerr_lst is None:
            yerr_lst = np.ones_like(y_lst)
        self.yerr_lst = yerr_lst
        self.norm     = norm
        # find weight list
        w = 1./self.yerr_lst**2
        self.w_lst = w/w.sum(dtype=np.float64)

        self.N = len(self.t_lst)
        #self.M = self.N/2.
        mint = np.abs(np.diff(t_lst)).min()
        tspan = t_lst.max()-t_lst.min()
        self.M = tspan/mint/2.

    def get_power(self, period):
        '''Get power spectrum.
        Args:
            period (:class:`numpy.array`): Period list to calculate the power
                spectrum.
        Returns:
            tuple: A tuple containing:
                * power_lst (:class:`numpy.array`): Powers at given periods.
                * winpower_lst (:class:`numpy.array`): Powers of the spectrum of
                  the window function at given periods.
        '''
        omega_lst = 2*math.pi/period

        power_lst = []
        window_power_lst = []

        # variable alias
        t = self.t_lst
        y = self.y_lst
        w = self.w_lst

        Y = (w*y).sum(dtype=np.float64)         # eq. (7)
        YYhat = (w*y**2).sum(dtype=np.float64)  # eq. (10) right
        YY = YYhat - Y**2                       # eq. (10) left
        for omega in omega_lst:
            x = omega*t
            cost = np.cos(x)
            sint = np.sin(x)
            C = (w*cost).sum(dtype=np.float64) # eq. (8)
            S = (w*sint).sum(dtype=np.float64) # eq. (9)
            YChat = (w*y*cost).sum(dtype=np.float64)    # eq. (11) right
            YShat = (w*y*sint).sum(dtype=np.float64)    # eq. (12) right
            CChat = (w*cost**2).sum(dtype=np.float64)   # eq. (13) right
            SShat = (w*sint**2).sum(dtype=np.float64)   # eq. (14) right
            #SShat = 1.0 - CChat
            CShat = (w*cost*sint).sum(dtype=np.float64) # eq. (15) right
            YC = YChat - Y*C    # eq. (11) left
            YS = YShat - Y*S    # eq. (12) left
            CC = CChat - C*C    # eq. (13) left
            SS = SShat - S*S    # eq. (14) left
            CS = CShat - C*S    # eq. (15) left
            D  = CC*SS - CS**2  # eq. (6)

            power = (SS*YC**2 + CC*YS**2 - 2.*CS*YC*YS)/YY/D
            power_lst.append(power)

            window_power = (cost.sum(dtype=np.float64))**2 + \
                           (sint.sum(dtype=np.float64))**2
            window_power_lst.append(window_power)

        power_lst = np.array(power_lst)

        window_power_lst = np.array(window_power_lst)/self.N**2

        if self.norm == 'none':
            return power_lst, window_power_lst
        elif self.norm == 'HorneBaliunas':
            factor = (self.N-1.)/2.
            return power_lst*factor, window_power_lst*factor
        elif self.norm == 'Cumming':
            factor = (self.N-3.)/2./(1.-power_lst.max())
            return power_lst*factor, window_power_lst*factor
        else:
            return power_lst, window_power_lst

    def power_to_prob(self, power):
        '''Convert power to probability.
        
        Args:
            power (float): Power
        Returns:
            float: Probability
        '''
        if self.norm == 'none':
            return math.exp(-power)
        elif self.norm == 'HorneBaliunas':
            return (1.-2.*power/(self.N-1.))**((self.N-3.)/2.)
        elif self.norm == 'Cumming':
            return (1.+2.*power/(self.N-3.))**(-(self.N-3.)/2.)
        else:
            return math.exp(-power)

    def prob_to_power(self, prob):
        '''Convert probability to power.
        Args:
            prob (float): Probability
        Returns:
            power (float): Power
        '''
        if self.norm == 'none':
            return -math.log(prob)
        elif self.norm == 'HorneBaliunas':
            return (self.N-1.)/2.*(1.-prob**(2./(self.N-3.)))
        elif self.norm == 'Cumming':
            return (self.N-3.)/2.*(prob**(-2./(self.N-3.))-1.)
        else:
            return -math.log(prob)

    def power_to_fap(self, power):
        '''Convert power to false alarm probability (FAP).
        
        Args:
            power (float): Power
        Returns:
            float: False alarm probability
        '''
        prob = self.power_to_prob(power)
        #prob = self.M*self.get_prob(power)
        return 1.-(1.-prob)**self.M

    def fap_to_power(self, fap):
        '''Convert false alarm probability (FAP) to power.
        
        Args:
            fap (float): False alarm probability
        Returns:
            float: Power
        '''
        prob = 1.-(1.-fap)**(1./self.M)
        return self.prob_to_power(prob)
