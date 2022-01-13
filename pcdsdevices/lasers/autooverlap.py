"""
Auto-overlap algorithm
"""

import logging

import time

import numpy as np

from pcdsdevices.pseudopos import DelayMotor
from pcdsdevices.epics_motor import SmarAct
from ophyd import EpicsSignal, EpicsSignalRO


class PFTSAutoOverlap:
    """
    Class for managing the automatic overlap of PFTS TCBOC signals.
    """
    def __init__(self, dly_line_pv, mon1_pv, mon2_pv, err_pv, buff_pv, rng_lo,
                 rng_hi):
        self._mon1 = EpicsSignalRO(mon1_pv, kind='normal')
        self._mon2 = EpicsSignalRO(mon2_pv, kind='normal')
        self._mons = [self._mon1, self._mon2]

        self._err = EpicsSignalRO(err_pv, kind='normal')

        self._buff = EpicsSignalRO(buff_pv, kind='normal')

        self._delay_motor = DelayMotor(SmarAct(dly_line_pv, name='motor'),
                                       n_bounces=4)

        self._search_low = rng_lo
        self._search_high = rng_hi

        logging.debug("init PFTSAutoOverlap")
        logging.debug("mon1_pv: {}".format(mon1_pv))
        logging.debug("mon2_pv: {}".format(mon2_pv))
        logging.debug("err_pv: {}".format(err_pv))
        logging.debug("dly_line_pv: {}".format(dly_line_pv))


    def _check_signals(self):
        """
        Verify that the digitizer signals are OK.
        """
        logging.debug("Running _check_signals")
        print("Checking digitizer signal status...")  # for now
        print("Signal status good.")  # for now
        return True


    def _check_motor(self):
        """
        Verify that the motor is OK.
        """
        logging.debug("Running _check_motor")
        print("Checking motor status...")
        print("Motor status good.")  # for now
        return True


    def _test_monitor(self, signal, threshold=35000):
        """
        Test a monitor signal to see if it's within the desired range. 

        Parameters
        ----------

        signal : array
            The signal to test.

        threshold : int
            The value, in ADC counts, that the maximum value of the
            waveform must exceed.
        """
        # Can make this more complicated in the future
        logging.debug("Running _test_monitor")
        mx = max(signal)
        logging.debug("threshold: {0}, max: {1}".format(threshold, mx))
        if mx > threshold:
            logging.debug("max > threshold")
            return True
        else:
            logging.debug("max < threshold")
            return False


    def _get_signal(self, signal, length):
        """
        Retrieve a signal from a Wave8 digitizer channel. The wave8 returns
        arrays of length 256 no matter the buffer length, so we use this
        function to clean up the array returned from EPICS.

        Parameters
        ----------
        signal : EpicsSignal or EpicsSignalRO
            The signal to read via signal.get()

        length : int
            The length of useful data in the array.
        """
        logging.debug("Running _get_signal")
        logging.debug("signal: {}".format(signal))
        logging.debug("length: {}".format(length))

        sig = signal.get()
        if len(sig) < length:
            logging.error("signal {0} is < {1} samples".format(signal,
                                                               length))
            raise ValueError
        else:
            return sig[0:length]
    
    def _average_signal(self, signal, navg, length):
        """
        Average an array signal.

        Parameters
        ----------
        signal : EpicsSignal or EpicsSignalRO
            The signal to read via signal.get()

        length : int
            The length of useful data in the array.

        navg : int
            The number of times to average the signal.
        """
        #TODO: Do this in a smarter way. Ensure that the wave8 is collecting
        # data properly, and collect averaged data at the same rate as the 
        # wave8 is returning data via EPICS. Just assume 10Hz for now.  
        logging.debug("Running _average_signal")
        logging.debug("signal: {}".format(signal))
        logging.debug("navg: {}".format(navg))
        logging.debug("length: {}".format(length))

        waveforms = []
        for i in range(navg):
            waveforms.append(self._get_signal(signal, length))
            time.sleep(0.1)
        return np.mean(list(zip(*waveforms)), axis=1)
        

    def optimize_monitor(self, stepsize=1e-12, threshold=35000, mon=0, navg=10):
        """
        Optimize the signal from the TCBOC monitor channels. This consists of
        ensuring that the signals are above a given threshold.

        Parameters
        ----------
        stepsize : float
            The step size, in picoseconds, of the delay motor for optimization.

        threshold : int
            The value, in ADC counts, that the maximum value of the
            waveform must exceed.
        
        mon : int
            The monitor signal to optimize. Currently accepts 0 or 1 (mon1 or
            mon2). 

        navg : int
            The number of waveforms to average before testing the threshold.
        """
        logging.debug("Running optimize_monitor")
        logging.debug("stepsize: {}".format(stepsize))
        logging.debug("signal: {}".format(threshold))
        logging.debug("mon: {}".format(mon))
        logging.debug("navg: {}".format(navg))

        motor_start = self._delay_motor.position.delay
        if motor_start < self._search_high and motor_start > self._search_low:
            direction = 1 # start in positive direction 
        else:
            s = "Delay motor is outside of {} to {}".format(self._search_low,
                                                            self._search_high)
            raise ValueError(s)

        while self._delay_motor.position.delay+stepsize < self._search_high:
            avg = self._average_signal(self._mons[mon], navg, self._buff.get())
            if self._test_monitor(avg, threshold):
                return True  # search finished
            self._delay_motor.mvr(stepsize*direction, wait=True)

        # Now try in reverse direction
        direction = -1
        while self._delay_motor.position.delay-stepsize > self._search_low:
            avg = self._average_signal(self._mons[mon], navg, self._buff.get())
            if self._test_monitor(avg, threshold):
                return True  # search finished
            self._delay_motor.mvr(stepsize*direction, wait=True)

        # If we got here, both directions failed. Return to start.
        self._delay_motor.mv(motor_start, wait=True)
        return False


    def _integrate_error(self, nbaseline=10, navg=10):
        avg = self._average_signal(self._err, navg, self._buff.get())
        baseline = np.average(avg[0:nbaseline])
        signal = [sig - baseline for sig in avg]
        return np.trapz(signal)
        
    def _measure_error(self, stepsize=0.05e-12, nbaseline=10, navg=20,
                       plot=False):
        """
        Measure the error signal (S-curve) from the TCBOC at different delays.
        """
        # The working range is ~2 ps. Assume that optimizing monitors got
        # close-ish to the center. So we move backwards 1 ps, then move 2 ps
        # in increments, measuring motor delay position vs integrated error.
        # Go to the minimum error (find zero crossing?). 
        motor_start = self._delay_motor.position.delay
        scan_start = motor_start - 1e-12
        scan_end = scan_start + 2e-12

        self._delay_motor.mv(scan_start, wait=True)
        delays = []
        errors = []
        while self._delay_motor.position.delay < scan_end:
            delays.append(self._delay_motor.position.delay)
            error = self._integrate_error(nbaseline=nbaseline, navg=navg)
            errors.append(error)
            self._delay_motor.mvr(stepsize, wait=True)

        if plot:
            import matplotlib.pyplot as plt
            plt.scatter(delays, errors)
            plt.show()
        return delays, errors
           

    def _find_zero_error(self, delays, errors, plot=False):
        """
        Find the delay that minimizes the integrated error curve generated by
        _measure_error.
        """
        # use derivative of error to find highest slope (most sensitive area)
        grad = np.gradient(errors)
        center = grad.argmax()  # index of highest gradient

        # Try linear fit around this area, about +/- 3 points
        spread = 3
        if center + spread > len(errors):
            end = len(errors)
        else:
            end = center + spread 

        if center - spread < 0:
            start = 0
        else:
            start = center - spread

        x = delays[start:end]
        y = errors[start:end]

        coeff = np.polyfit(x, y, 1)

        zero = -coeff[1]/coeff[0]

        # All the edge cases I can think of right now
        if zero < delays[0] or zero > delays[-1]:  # Something went wrong
            raise ValueError("Zero outside of measured range! {}".format(zero)) 
        elif zero < 0:
            raise ValueError("Zero is negative! {}".format(zero))

        if plot:
            fit_fn = np.poly1d(coeff)
            import matplotlib.pyplot as plt
            plt.plot(x, y, 'ro', x, fit_fn(x), '--k')
            plt.show()

        return zero


    def auto_overlap(self, mon_thresh=35000, coarse_step=1e-12,
                     fine_step=0.01e-12, nbaseline=10, navg=20):
        self._check_signals()
        self._check_motor()

        i = 0
        while i < 3:  # Should only take one try
            mon1 =  self.optimize_monitor(stepsize=coarse_step,
                                          threshold=mon_thresh, mon=0,
                                          navg=navg)
            mon2 =  self.optimize_monitor(stepsize=coarse_step,
                                          threshold=mon_thresh, mon=1,
                                          navg=navg)

            test_1 = self._test_monitor(self._mon1.get(), threshold=mon_thresh)
            test_2 = self._test_monitor(self._mon2.get(), threshold=mon_thresh)

            if test_1 and test_2: break

            i += 1
        else:
            raise Exception("Unable to optimize monitor signals!")

        
        delays, errors = self._measure_error(stepsize=fine_step,
                                             nbaseline=nbaseline, navg=navg,
                                             plot=False)

        zero = self._find_zero_error(delays, errors, plot=False)

        self._delay_motor.mv(zero, wait=True)
