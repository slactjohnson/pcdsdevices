"""
Module to define `Slits` classes.

The SLAC EPICS motor record contains an extra set of records to abstract four
axes into a Slits object. This allows an operator to manipulate the center and
width in two dimensions of a small aperture. The classes below allow both
individual parameters of the aperture and the Slit as a whole to be controlled
and scanned. The `Slits` class instantiates four sub-devices: `~Slits.xwidth`,
`~Slits.xcenter`, `~Slits.ycenter`, `~Slits.ywidth`. These are each represented
by a `SlitPositioner`. The main `Slits` class assumes that most of
the manipulation will be done on the size of the aperture not the position,
however, if control of the center is desired the ``center`` sub-devices can be
used.
"""
import logging
from collections import OrderedDict

from ophyd import Component as Cpt
from ophyd import Device
from ophyd import DynamicDeviceComponent as DDCpt
from ophyd import EpicsSignal, EpicsSignalRO
from ophyd import FormattedComponent as FCpt
from ophyd.pv_positioner import PVPositioner
from ophyd.signal import Signal
from ophyd.status import wait as status_wait, Status

from .epics_motor import BeckhoffAxis
from .interface import FltMvInterface, MvInterface, LightpathMixin
from .signal import PytmcSignal, NotImplementedSignal
from .sensors import RTD
from .utils import schedule_task, get_status_value

logger = logging.getLogger(__name__)


class SlitsBase(MvInterface, Device, LightpathMixin):
    """
    Base class for slit motion interfacing.
    """
    # QIcon for UX
    _icon = 'fa.th-large'

    # Mark as parent class for lightpath interface
    _lightpath_mixin = True
    lightpath_cpts = ['xwidth', 'ywidth']

    # Tab settings
    tab_whitelist = ['open', 'close', 'block']

    # Just to hold a value
    nominal_aperture = Cpt(Signal, kind='normal')

    # Placeholders for each component to override
    # These are expected to be positioners
    xwidth = None
    ywidth = None
    xcenter = None
    ycenter = None

    def __init__(self, *args, nominal_aperture=5.0, **kwargs):
        self._has_subscribed = False
        super().__init__(*args, **kwargs)
        self.nominal_aperture.put(nominal_aperture)

    def format_status_info(self, status_info):
        """
        Override status info handler to render the slits.

        Display slits status info in the ipython terminal.

        Parameters
        ----------
        status_info: dict
            Nested dictionary. Each level has keys name, kind, and is_device.
            If is_device is True, subdevice dictionaries may follow. Otherwise,
            the only other key in the dictionary will be value.

        Returns
        -------
        status: str
            Formatted string with all relevant status information.
        """
        # happi metadata
        try:
            md = self.root.md
        except AttributeError:
            name = f'Slit: {self.prefix}'
        else:
            beamline = get_status_value(md, 'beamline')
            stand = get_status_value(md, 'stand')
            if stand is not None:
                name = f'{beamline} Slit {self.name} on {stand}'
            else:
                name = f'{beamline} Slit {self.name}'

        x_width = get_status_value(status_info, 'xwidth', 'position')
        y_width = get_status_value(status_info, 'ywidth', 'position')
        x_center = get_status_value(status_info, 'xcenter', 'position')
        y_center = get_status_value(status_info, 'ycenter', 'position')
        w_units = get_status_value(status_info, 'ywidth', 'setpoint', 'units')
        c_units = get_status_value(status_info, 'ycenter', 'setpoint', 'units')

        return f"""\
{name}
(hg, vg): ({x_width:+.4f}, {y_width:+.4f}) [{w_units}]
(ho, vo): ({x_center:+.4f}, {y_center:+.4f}) [{c_units}]
"""

    def move(self, size, wait=False, moved_cb=None, timeout=None):
        """
        Set the dimensions of the width/height of the gap to width paramater.

        Parameters
        ---------
        size : float or tuple of float
            Target size for slits in both x and y axis. Either specify as a
            tuple for a rectangular aperture, ``(width, height)``, or set both
            with single floating point value to use a square opening.

        wait : bool
            If `True`, block until move is completed.

        timeout : float, optional
            Maximum time for the motion. If `None` is given, the default value
            of `xwidth` and `ywidth` positioners is used.

        moved_cb : callable, optional
            Function to be run when the operation finishes. This callback
            should not expect any arguments or keywords.

        Returns
        -------
        status : AndStatus
            Logical combination of the request to both horizontal and vertical
            motors.
        """

        # Check for rectangular setpoint
        if isinstance(size, tuple):
            (width, height) = size
        else:
            width, height = size, size
        # Instruct both width and height then combine the output status
        x_stat = self.xwidth.move(width, wait=False, timeout=timeout)
        y_stat = self.ywidth.move(height, wait=False, timeout=timeout)
        status = x_stat & y_stat
        # Add our callback if one was given
        if moved_cb is not None:
            status.add_callback(moved_cb)
        # Wait if instructed to do so. Stop the motors if interrupted
        if wait:
            try:
                status_wait(status)
            except KeyboardInterrupt:
                self.xwidth.stop()
                self.ywidth.stop()
                raise
        return status

    @property
    def current_aperture(self):
        """
        Current size of the aperture. Returns a tuple in the form
        ``(width, height)``.
        """
        return (self.xwidth.position, self.ywidth.position)

    @property
    def position(self):
        return self.current_aperture

    def remove(self, size=None, wait=False, timeout=None, **kwargs):
        """
        Open the slits to unblock the beam.

        Parameters
        ----------
        size : float, optional
            Open the slits to a specific size. Defaults to `.nominal_aperture`.

        wait : bool, optional
            Wait for the status object to complete the move before returning.

        timeout : float, optional
            Maximum time to wait for the motion. If `None`, the default timeout
            for this positioner is used.

        Returns
        -------
        Status
            `~ophyd.Status` object based on move completion.

        See Also
        --------
        :meth:`Slits.move`
        """

        # Use nominal_aperture by default
        size = size or self.nominal_aperture
        if size > min(self.current_aperture):
            return self.move(size, wait=wait, timeout=timeout, **kwargs)
        else:
            status = Status()
            status.set_finished()
            return status

    def set(self, size):
        """Alias for the move method, here for ``bluesky`` compatibilty."""
        return self.move(size, wait=False)

    def stage(self):
        """
        Store the initial values of the aperture position before scanning.
        """
        self._original_vals[self.xwidth.setpoint] = self.xwidth.readback.get()
        self._original_vals[self.ywidth.setpoint] = self.ywidth.readback.get()
        return super().stage()

    def subscribe(self, cb, event_type=None, run=True):
        """
        Subscribe to changes of the slits.
        Parameters
        ----------
        cb : callable
            Callback to be run.
        event_type : str, optional
            Type of event to run callback on.
        run : bool, optional
            Run the callback immediately.
        """

        # Avoid making child subscriptions unless a client cares
        if not self._has_subscribed:
            # Subscribe to changes in aperture
            self.xwidth.readback.subscribe(self._aperture_changed,
                                           run=False)
            self.ywidth.readback.subscribe(self._aperture_changed,
                                           run=False)
            self._has_subscribed = True
        return super().subscribe(cb, event_type=event_type, run=run)

    def _aperture_changed(self, *args, **kwargs):
        """Callback run when slit size is adjusted."""
        # Avoid duplicate keywords
        kwargs.pop('sub_type', None)
        kwargs.pop('obj',      None)
        # Run subscriptions
        self._run_subs(sub_type=self.SUB_STATE, obj=self, **kwargs)

    def _set_lightpath_states(self, lightpath_values):
        widths = [kw['value'] for kw in lightpath_values.values()]
        self._inserted = (min(widths) < self.nominal_aperture.get())
        self._removed = not self._inserted


class BadSlitPositionerBase(FltMvInterface, PVPositioner):
    """Base class for slit positioner with awful PV names."""

    readback = FCpt(EpicsSignalRO, '{prefix}:ACTUAL_{_dirlong}',
                    auto_monitor=True, kind='normal')
    setpoint = FCpt(EpicsSignal, '{prefix}:{_dirshort}_REQ',
                    auto_monitor=True, kind='normal')

    def __init__(self, prefix, *, slit_type="", limits=None, **kwargs):
        # Private PV names to deal with complex naming schema
        self._dirlong = slit_type
        self._dirshort = slit_type[:4]
        # Initalize PVPositioner
        super().__init__(prefix, limits=limits, **kwargs)


class LusiSlitPositioner(BadSlitPositionerBase):
    """
    Abstraction of a Slit axis from LCLS-I

    Each adjustable parameter of the slit (center, width) can be modeled as a
    motor in itself, even though each controls two different actual motors in
    reality, this gives a convienent interface for adjusting the aperture size
    and location with out backwards calculating motor positions.

    Parameters
    ----------
    prefix : str
        The prefix location of the slits, e.g. 'MFX:DG2'.

    name : str
        Alias for the axis.

    slit_type : {'XWIDTH', 'YWIDTH', 'XCENTER', 'YCENTER'}
        The aspect of the slit position you would like to control with this
        specific motor.

    limits : tuple, optional
        Limits on the motion of the positioner. By default, the limits on the
        setpoint PV are used if `None` is given.

    See Also
    --------
    `ophyd.PVPositioner`
        SlitPositioner inherits directly from `~ophyd.PVPositioner`.
    """

    done = Cpt(EpicsSignalRO, ':DMOV', auto_monitor=True, kind='omitted')

    @property
    def egu(self):
        """Engineering Units."""
        return self._egu or self.readback._read_pv.units

    def _setup_move(self, position):
        # This is subclassed because we need `wait` to be set to `False` unlike
        # the default PVPositioner method. `wait` set to `True` will not return
        # until the move has completed
        logger.debug('%s.setpoint = %s', self.name, position)
        self.setpoint.put(position, wait=False)


class SlitPositioner(LusiSlitPositioner):
    # Should probably deprecate this name
    pass


class LusiSlits(SlitsBase):
    """
    Beam slits with combined motion for center and width.

    Parameters
    ----------
    prefix : str
        The EPICS base PV of the motor.

    name : str, optional
        The name of the offset mirror.

    nominal_aperture : float, optional
        Nominal slit size that will encompass the beam without blocking.

    Notes
    -----
    The slits represent a unique device when forming the lightpath because
    whether the beam is being blocked or not depends on the pointing. In order
    to create an estimate that will warn operators of 'closed' slits, we set a
    `nominal_aperture` for each unique device along the beamline. This is
    value is considered the smallest the slit aperture can become without
    blocking the beamline. Both the `xwidth` and the `ywidth`(height) need to
    exceed this `nominal_aperture` for the slits to be considered removed.
    """

    # Base class overrides
    xwidth = Cpt(LusiSlitPositioner, '', slit_type='XWIDTH', kind='hinted')
    ywidth = Cpt(LusiSlitPositioner, '', slit_type='YWIDTH', kind='hinted')
    xcenter = Cpt(LusiSlitPositioner, '', slit_type='XCENTER', kind='normal')
    ycenter = Cpt(LusiSlitPositioner, '', slit_type='YCENTER', kind='normal')

    # Local PVs
    blocked = Cpt(EpicsSignalRO, ':BLOCKED', kind='omitted')
    open_cmd = Cpt(EpicsSignal, ':OPEN', kind='omitted')
    close_cmd = Cpt(EpicsSignal, ':CLOSE', kind='omitted')
    block_cmd = Cpt(EpicsSignal, ':BLOCK', kind='omitted')

    def open(self):
        """Uses the built-in 'OPEN' record to move open the aperture."""
        self.open_cmd.put(1)

    def close(self):
        """Close the slits to have an aperture of 0mm on each side."""
        self.close_cmd.put(1)

    def block(self):
        """Overlap the slits to block the beam."""
        self.block_cmd.put(1)


class Slits(LusiSlits):
    # Should Probably Deprecate this name
    pass


class BeckhoffSlitPositioner(BadSlitPositionerBase):
    """
    Abstraction of a Slit axis from LCLS-II.

    This class needs a BeckhoffSlits parent to function properly.
    """

    readback = FCpt(PytmcSignal, BadSlitPositionerBase.readback.suffix,
                    io='i', auto_monitor=True, kind='normal')
    setpoint = FCpt(PytmcSignal, BadSlitPositionerBase.setpoint.suffix,
                    io='io', auto_monitor=True, kind='normal')
    done = Cpt(Signal, kind='omitted')
    actuate = Cpt(Signal, kind='omitted')

    @actuate.sub_value
    def _execute_move(self, *args, value, **kwargs):
        if value == 1:
            self.parent.exec_queue.put(1)

    @done.sub_value
    def _reset_actuate(self, *args, value, old_value, **kwargs):
        if value == 1 and old_value == 0:
            self.actuate.put(0)


class BeckhoffSlits(SlitsBase):
    # Base class overrides
    xwidth = Cpt(BeckhoffSlitPositioner, '', slit_type='XWIDTH', kind='hinted')
    ywidth = Cpt(BeckhoffSlitPositioner, '', slit_type='YWIDTH', kind='hinted')
    xcenter = Cpt(BeckhoffSlitPositioner, '', slit_type='XCENTER',
                  kind='normal')
    ycenter = Cpt(BeckhoffSlitPositioner, '', slit_type='YCENTER',
                  kind='normal')

    # Slit state commands
    exec_queue = Cpt(Signal, kind='omitted')
    exec_move = Cpt(PytmcSignal, ':GO', io='io', kind='omitted')

    # Slit calculated move dmov
    done_all = Cpt(Signal, kind='omitted')
    done_top = Cpt(PytmcSignal, ':TOP:DMOV', io='i', kind='omitted')
    done_bottom = Cpt(PytmcSignal, ':BOTTOM:DMOV', io='i', kind='omitted')
    done_north = Cpt(PytmcSignal, ':NORTH:DMOV', io='i', kind='omitted')
    done_south = Cpt(PytmcSignal, ':SOUTH:DMOV', io='i', kind='omitted')

    # Raw motors
    top = Cpt(BeckhoffAxis, ':MMS:TOP', kind='normal')
    bottom = Cpt(BeckhoffAxis, ':MMS:BOTTOM', kind='normal')
    north = Cpt(BeckhoffAxis, ':MMS:NORTH', kind='normal')
    south = Cpt(BeckhoffAxis, ':MMS:SOUTH', kind='normal')

    def __init__(self, prefix, *, name, **kwargs):
        self._started_move = False
        super().__init__(prefix, name=name, **kwargs)

    @exec_queue.sub_value
    def _exec_handler(self, *args, value, old_value, **kwargs):
        """Wait just a moment to queue up move requests."""
        if value == 1 and old_value == 0:
            self._started_move = True
            schedule_task(self.exec_move.put, args=(1,), delay=0.2)

    @done_all.sub_value
    def _reset_exec_move(self, *args, value, old_value, **kwargs):
        """When we're done moving, reset the exec_move signal."""
        if self._started_move and value == 1 and old_value == 0:
            self._started_move = False
            self.exec_queue.put(0)
            self.exec_move.put(0)

    @done_all.sub_value
    def _dmov_fanout(self, *args, value, **kwargs):
        """When we're done moving, tell our pv positioners."""
        self.xwidth.done.put(value)
        self.ywidth.done.put(value)
        self.xcenter.done.put(value)
        self.ycenter.done.put(value)

    @done_top.sub_value
    @done_bottom.sub_value
    @done_north.sub_value
    @done_south.sub_value
    def _update_dmov(self, *args, **kwargs):
        """When part of the dmov updates, update the done_all flag."""
        done = all((self.done_top.get(),
                    self.done_bottom.get(),
                    self.done_north.get(),
                    self.done_south.get()))
        if done != self.done_all.get():
            self.done_all.put(done)


def _rtd_fields(cls, attr_base, range_, **kwargs):
    padding = max(range_)//10 + 2
    defn = OrderedDict()
    for i in range_:
        attr = '{attr}{i}'.format(attr=attr_base, i=i)
        suffix = ':RTD:{i}'.format(i=str(i).zfill(padding))
        defn[attr] = (cls, suffix, kwargs)
    return defn


class PowerSlits(BeckhoffSlits):
    """
    'SL*:POWER'.

    Power slits variant of slits. The XTES design.

    Parameters
    ----------
    prefix : str
        The PV base of the device.
    """

    rtds = DDCpt(_rtd_fields(RTD, 'rtd', range(1, 9)))
    fsw = Cpt(NotImplementedSignal, ':FSW', kind='normal')
