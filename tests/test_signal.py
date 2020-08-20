import logging
import threading
from unittest.mock import Mock

from ophyd.signal import EpicsSignal, EpicsSignalRO, Signal
from ophyd.sim import FakeEpicsSignal
from pcdsdevices.signal import (AvgSignal, PytmcSignal,
                                UnitConversionDerivedSignal)

logger = logging.getLogger(__name__)


def test_pytmc_signal():
    logger.debug('test_pytmc_signal')
    # Just make sure the normal use cases aren't super broken
    rwsig = PytmcSignal('PREFIX', io='io')
    rosig = PytmcSignal('PREFIX', io='i')
    assert isinstance(rwsig, EpicsSignal)
    assert isinstance(rwsig, PytmcSignal)
    assert isinstance(rosig, EpicsSignalRO)
    assert isinstance(rosig, PytmcSignal)


def test_avg_signal():
    logger.debug('test_avg_signal')
    sig = Signal(name='raw')
    avg = AvgSignal(sig, 2, name='avg')

    assert avg.averages == 2

    sig.put(1)
    assert avg.get() == 1
    sig.put(3)
    assert avg.get() == 2
    sig.put(2)
    assert avg.get() == 2.5

    avg.averages = 3

    sig.put(1)
    assert avg.get() == 1
    sig.put(3)
    assert avg.get() == 2
    sig.put(2)
    assert avg.get() == 2

    cb = Mock()
    avg.subscribe(cb)
    sig.put(0)
    assert cb.called


def test_unit_conversion_signal():
    orig = FakeEpicsSignal('sig', name='orig')
    orig.sim_put(5)

    converted = UnitConversionDerivedSignal(
        derived_from=orig,
        original_units='m',
        derived_units='mm',
        name='converted',
    )

    assert converted.original_units == 'm'
    assert converted.derived_units == 'mm'
    assert converted.describe()[converted.name]['units'] == 'mm'

    assert converted.get() == 5_000
    converted.put(10_000, wait=True)
    assert orig.get() == 10

    event = threading.Event()
    cb = Mock()

    def callback(**kwargs):
        cb(**kwargs)
        event.set()

    converted.subscribe(callback, run=False)
    orig.put(20, wait=True)
    event.wait(1)
    cb.assert_called_once()

    args, kwargs = cb.call_args
    assert kwargs['value'] == 20_000
    assert converted.get() == 20_000
