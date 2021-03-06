import inspect
import os
import pkgutil
import shutil
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import ophyd
import pytest
from epics import PV
from ophyd.signal import LimitError
from ophyd.sim import FakeEpicsSignal, make_fake_device

from pcdsdevices.attenuator import MAX_FILTERS, Attenuator, _att_classes
from pcdsdevices.mv_interface import setup_preset_paths

MODULE_PATH = Path(__file__).parent


# Signal.put warning is a testing artifact.
# FakeEpicsSignal needs an update, but I don't have time today
# Needs to not pass tons of kwargs up to Signal.put
warnings.filterwarnings('ignore',
                        message='Signal.put no longer takes keyword arguments')


# Other temporary patches to FakeEpicsSignal
def check_value(self, value):
    if value is None:
        raise ValueError('Cannot write None to epics PVs')
    if not self._use_limits:
        return

    low_limit, high_limit = self.limits
    if low_limit >= high_limit:
        return

    if not (low_limit <= value <= high_limit):
        raise LimitError('Value {} outside of range: [{}, {}]'
                         .format(value, low_limit, high_limit))


# Check value is busted, ignores (0, 0) no limits case
FakeEpicsSignal.check_value = check_value
# Metadata changed is missing
FakeEpicsSignal._metadata_changed = lambda *args, **kwargs: None
# pvname is missing
FakeEpicsSignal.pvname = ''
# lots of things are missing
FakeEpicsSignal._read_pv = SimpleNamespace(get_ctrlvars=lambda: None)
# Stupid patch that somehow makes the test cleanup bug go away
PV.count = property(lambda self: 1)

for name, cls in _att_classes.items():
    _att_classes[name] = make_fake_device(cls)


# Used in multiple test files
@pytest.fixture(scope='function')
def fake_att():
    att = Attenuator('TST:ATT', MAX_FILTERS-1, name='test_att')
    att.readback.sim_put(1)
    att.done.sim_put(0)
    att.calcpend.sim_put(0)
    for i, filt in enumerate(att.filters):
        filt.state.put('OUT')
        filt.thickness.put(2*i)
    return att


@pytest.fixture(scope='function')
def presets():
    folder_obj = Path(__file__).parent / 'test_presets'
    folder = str(folder_obj)
    if folder_obj.exists():
        shutil.rmtree(folder)
    hutch = folder + '/hutch'
    user = folder + '/user'
    os.makedirs(hutch)
    os.makedirs(user)
    setup_preset_paths(hutch=hutch, user=user)
    yield
    setup_preset_paths()
    shutil.rmtree(folder)


def find_all_device_classes() -> list:
    exclude_list = {'_version', }
    pkgname = 'pcdsdevices'
    modules = [
        mod.name for mod in pkgutil.iter_modules(
            path=[MODULE_PATH.parent / pkgname])
        if mod not in exclude_list
    ]

    for module in modules:
        __import__(f'{pkgname}.{module}')

    devices = set()
    for mod_name, mod in sys.modules.items():
        if pkgname not in mod_name:
            continue

        for mod_attr in dir(mod):
            obj = getattr(mod, mod_attr)
            if inspect.isclass(obj) and issubclass(obj, ophyd.Device):
                if not obj.__module__.startswith('ophyd'):
                    devices.add(obj)

    return list(devices)
