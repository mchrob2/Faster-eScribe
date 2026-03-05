# SPDX-FileCopyrightText: 2017 Tony DiCola for Adafruit Industries
#
# SPDX-License-Identifier: MIT

"""
`adafruit_rfm69`
====================================================
CircuitPython RFM69 packet radio module.
Optimized for high‑rate streaming.
"""

import random
import time

import adafruit_bus_device.spi_device as spidev
from micropython import const

HAS_SUPERVISOR = False
try:
    import supervisor
    HAS_SUPERVISOR = hasattr(supervisor, "ticks_ms")
except ImportError:
    pass

try:
    from typing import Callable, Optional, Type
    from busio import SPI
    from circuitpython_typing import ReadableBuffer, WriteableBuffer
    from digitalio import DigitalInOut
except ImportError:
    pass

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/adafruit/Adafruit_CircuitPython_RFM69.git"

# Internal constants:
_REG_FIFO = const(0x00)
_REG_OP_MODE = const(0x01)
_REG_DATA_MOD = const(0x02)
_REG_BITRATE_MSB = const(0x03)
_REG_BITRATE_LSB = const(0x04)
_REG_FDEV_MSB = const(0x05)
_REG_FDEV_LSB = const(0x06)
_REG_FRF_MSB = const(0x07)
_REG_FRF_MID = const(0x08)
_REG_FRF_LSB = const(0x09)
_REG_VERSION = const(0x10)
_REG_PA_LEVEL = const(0x11)
_REG_OCP = const(0x13)
_REG_RX_BW = const(0x19)
_REG_AFC_BW = const(0x1A)
_REG_RSSI_VALUE = const(0x24)
_REG_DIO_MAPPING1 = const(0x25)
_REG_IRQ_FLAGS1 = const(0x27)
_REG_IRQ_FLAGS2 = const(0x28)
_REG_PREAMBLE_MSB = const(0x2C)
_REG_PREAMBLE_LSB = const(0x2D)
_REG_SYNC_CONFIG = const(0x2E)
_REG_SYNC_VALUE1 = const(0x2F)
_REG_PACKET_CONFIG1 = const(0x37)
_REG_FIFO_THRESH = const(0x3C)
_REG_PACKET_CONFIG2 = const(0x3D)
_REG_AES_KEY1 = const(0x3E)
_REG_TEMP1 = const(0x4E)
_REG_TEMP2 = const(0x4F)
_REG_TEST_PA1 = const(0x5A)
_REG_TEST_PA2 = const(0x5C)
_REG_TEST_DAGC = const(0x6F)

_TEST_PA1_NORMAL = const(0x55)
_TEST_PA1_BOOST = const(0x5D)
_TEST_PA2_NORMAL = const(0x70)
_TEST_PA2_BOOST = const(0x7C)
_OCP_NORMAL = const(0x1A)
_OCP_HIGH_POWER = const(0x0F)

_FXOSC = 32000000.0
_FSTEP = _FXOSC / 524288

_RH_BROADCAST_ADDRESS = const(0xFF)
_RH_FLAGS_ACK = const(0x80)
_RH_FLAGS_RETRY = const(0x40)

SLEEP_MODE = 0b000
STANDBY_MODE = 0b001
FS_MODE = 0b010
TX_MODE = 0b011
RX_MODE = 0b100

_TICKS_PERIOD = const(1 << 29)
_TICKS_MAX = const(_TICKS_PERIOD - 1)
_TICKS_HALFPERIOD = const(_TICKS_PERIOD // 2)

def ticks_diff(ticks1: int, ticks2: int) -> int:
    diff = (ticks1 - ticks2) & _TICKS_MAX
    diff = ((diff + _TICKS_HALFPERIOD) & _TICKS_MAX) - _TICKS_HALFPERIOD
    return diff

def check_timeout(flag: Callable, limit: float) -> bool:
    timed_out = False
    if HAS_SUPERVISOR:
        start = supervisor.ticks_ms()
        while not timed_out and not flag():
            if ticks_diff(supervisor.ticks_ms(), start) >= limit * 1000:
                timed_out = True
    else:
        start = time.monotonic()
        while not timed_out and not flag():
            if time.monotonic() - start >= limit:
                timed_out = True
    return timed_out


class RFM69:
    _BUFFER = bytearray(4)

    class _RegisterBits:
        def __init__(self, address: int, *, offset: int = 0, bits: int = 1) -> None:
            assert 0 <= offset <= 7
            assert 1 <= bits <= 8
            assert (offset + bits) <= 8
            self._address = address
            self._mask = 0
            for _ in range(bits):
                self._mask <<= 1
                self._mask |= 1
            self._mask <<= offset
            self._offset = offset

        def __get__(self, obj: Optional["RFM69"], objtype: Type["RFM69"]):
            reg_value = obj._read_u8(self._address)
            return (reg_value & self._mask) >> self._offset

        def __set__(self, obj: Optional["RFM69"], val: int) -> None:
            reg_value = obj._read_u8(self._address)
            reg_value &= ~self._mask
            reg_value |= (val & 0xFF) << self._offset
            obj._write_u8(self._address, reg_value)

    # Control bits
    data_mode = _RegisterBits(_REG_DATA_MOD, offset=5, bits=2)
    modulation_type = _RegisterBits(_REG_DATA_MOD, offset=3, bits=2)
    modulation_shaping = _RegisterBits(_REG_DATA_MOD, offset=0, bits=2)
    temp_start = _RegisterBits(_REG_TEMP1, offset=3)
    temp_running = _RegisterBits(_REG_TEMP1, offset=2)
    sync_on = _RegisterBits(_REG_SYNC_CONFIG, offset=7)
    sync_size = _RegisterBits(_REG_SYNC_CONFIG, offset=3, bits=3)
    aes_on = _RegisterBits(_REG_PACKET_CONFIG2, offset=0)
    pa_0_on = _RegisterBits(_REG_PA_LEVEL, offset=7)
    pa_1_on = _RegisterBits(_REG_PA_LEVEL, offset=6)
    pa_2_on = _RegisterBits(_REG_PA_LEVEL, offset=5)
    output_power = _RegisterBits(_REG_PA_LEVEL, offset=0, bits=5)
    rx_bw_dcc_freq = _RegisterBits(_REG_RX_BW, offset=5, bits=3)
    rx_bw_mantissa = _RegisterBits(_REG_RX_BW, offset=3, bits=2)
    rx_bw_exponent = _RegisterBits(_REG_RX_BW, offset=0, bits=3)
    afc_bw_dcc_freq = _RegisterBits(_REG_AFC_BW, offset=5, bits=3)
    afc_bw_mantissa = _RegisterBits(_REG_AFC_BW, offset=3, bits=2)
    afc_bw_exponent = _RegisterBits(_REG_AFC_BW, offset=0, bits=3)
    packet_format = _RegisterBits(_REG_PACKET_CONFIG1, offset=7, bits=1)
    dc_free = _RegisterBits(_REG_PACKET_CONFIG1, offset=5, bits=2)
    crc_on = _RegisterBits(_REG_PACKET_CONFIG1, offset=4, bits=1)
    crc_auto_clear_off = _RegisterBits(_REG_PACKET_CONFIG1, offset=3, bits=1)
    address_filter = _RegisterBits(_REG_PACKET_CONFIG1, offset=1, bits=2)
    mode_ready = _RegisterBits(_REG_IRQ_FLAGS1, offset=7)
    dio_0_mapping = _RegisterBits(_REG_DIO_MAPPING1, offset=6, bits=2)

    def __init__(
        self,
        spi: SPI,
        cs: DigitalInOut,
        reset: DigitalInOut,
        frequency: int,
        *,
        sync_word: bytes = b"\x2d\xd4",
        preamble_length: int = 4,
        encryption_key: Optional[bytes] = None,
        high_power: bool = True,
        baudrate: int = 2000000,
    ) -> None:
        self._tx_power = 13
        self.high_power = high_power
        self._device = spidev.SPIDevice(spi, cs, baudrate=baudrate, polarity=0, phase=0)
        self._reset = reset
        self._reset.switch_to_output(value=False)
        self.reset()
        version = self._read_u8(_REG_VERSION)
        if version not in {0x23, 0x24}:
            raise RuntimeError("Invalid RFM69 version, check wiring!")
        self.idle()
        self._write_u8(_REG_FIFO_THRESH, 0b10001111)  # default: TX condition = FIFO level, threshold=15
        self._write_u8(_REG_TEST_DAGC, 0x30)
        self.sync_word = sync_word
        self.preamble_length = preamble_length
        self.frequency_mhz = frequency
        self.encryption_key = encryption_key
        self.modulation_shaping = 0b01
        self.bitrate = 250000
        self.frequency_deviation = 250000
        self.rx_bw_dcc_freq = 0b111
        self.rx_bw_mantissa = 0b00
        self.rx_bw_exponent = 0b000
        self.afc_bw_dcc_freq = 0b111
        self.afc_bw_mantissa = 0b00
        self.afc_bw_exponent = 0b000
        self.packet_format = 1
        self.dc_free = 0b10
        self.tx_power = 13

        self.last_rssi = 0.0
        self.ack_wait = 0.5
        self.receive_timeout = 0.5
        self.xmit_timeout = 2.0          # can be reduced by application
        self.ack_retries = 5
        self.ack_delay = None
        self.sequence_number = 0
        self.seen_ids = bytearray(256)
        self.node = _RH_BROADCAST_ADDRESS
        self.destination = _RH_BROADCAST_ADDRESS
        self.identifier = 0
        self.flags = 0

    # -------------------------------------------------------------------------
    # New property: FIFO threshold (0–127). Top bit (TxStartCondition) preserved.
    @property
    def fifo_threshold(self) -> int:
        return self._read_u8(_REG_FIFO_THRESH) & 0x7F

    @fifo_threshold.setter
    def fifo_threshold(self, val: int) -> None:
        assert 0 <= val <= 0x7F
        # keep the top bit (TxStartCondition) unchanged
        reg = self._read_u8(_REG_FIFO_THRESH) & 0x80
        self._write_u8(_REG_FIFO_THRESH, reg | val)

    # -------------------------------------------------------------------------
    def _read_into(self, address: int, buf: WriteableBuffer, length: Optional[int] = None) -> None:
        if length is None:
            length = len(buf)
        with self._device as device:
            self._BUFFER[0] = address & 0x7F
            device.write(self._BUFFER, end=1)
            device.readinto(buf, end=length)

    def _read_u8(self, address: int) -> int:
        self._read_into(address, self._BUFFER, length=1)
        return self._BUFFER[0]

    def _write_from(self, address: int, buf: ReadableBuffer, length: Optional[int] = None) -> None:
        if length is None:
            length = len(buf)
        with self._device as device:
            self._BUFFER[0] = (address | 0x80) & 0xFF
            device.write(self._BUFFER, end=1)
            device.write(buf, end=length)

    def _write_u8(self, address: int, val: int) -> None:
        with self._device as device:
            self._BUFFER[0] = (address | 0x80) & 0xFF
            self._BUFFER[1] = val & 0xFF
            device.write(self._BUFFER, end=2)

    def reset(self) -> None:
        self._reset.value = True
        time.sleep(0.0001)
        self._reset.value = False
        time.sleep(0.005)

    def disable_boost(self) -> None:
        if self.high_power:
            self._write_u8(_REG_TEST_PA1, _TEST_PA1_NORMAL)
            self._write_u8(_REG_TEST_PA2, _TEST_PA2_NORMAL)
            self._write_u8(_REG_OCP, _OCP_NORMAL)

    def idle(self) -> None:
        self.disable_boost()
        self.operation_mode = STANDBY_MODE

    def sleep(self) -> None:
        self.operation_mode = SLEEP_MODE

    def listen(self) -> None:
        self.disable_boost()
        self.dio_0_mapping = 0b01
        self.operation_mode = RX_MODE

    def transmit(self) -> None:
        if self.high_power and (self._tx_power >= 18):
            self._write_u8(_REG_TEST_PA1, _TEST_PA1_BOOST)
            self._write_u8(_REG_TEST_PA2, _TEST_PA2_BOOST)
            self._write_u8(_REG_OCP, _OCP_HIGH_POWER)
        self.dio_0_mapping = 0b00
        self.operation_mode = TX_MODE

    @property
    def temperature(self) -> float:
        self.idle()
        self.temp_start = 1
        while self.temp_running > 0:
            pass
        temp = self._read_u8(_REG_TEMP2)
        return 166.0 - temp

    @property
    def operation_mode(self) -> int:
        op_mode = self._read_u8(_REG_OP_MODE)
        return (op_mode >> 2) & 0b111

    @operation_mode.setter
    def operation_mode(self, val: int) -> None:
        assert 0 <= val <= 4
        op_mode = self._read_u8(_REG_OP_MODE)
        op_mode &= 0b11100011
        op_mode |= val << 2
        self._write_u8(_REG_OP_MODE, op_mode)
        if HAS_SUPERVISOR:
            start = supervisor.ticks_ms()
            while not self.mode_ready:
                if ticks_diff(supervisor.ticks_ms(), start) >= 1000:
                    raise TimeoutError("Operation Mode failed to set.")
        else:
            start = time.monotonic()
            while not self.mode_ready:
                if time.monotonic() - start >= 1:
                    raise TimeoutError("Operation Mode failed to set.")

    @property
    def sync_word(self) -> Optional[bytearray]:
        if not self.sync_on:
            return None
        sync_word_length = self.sync_size + 1
        sync_word = bytearray(sync_word_length)
        self._read_into(_REG_SYNC_VALUE1, sync_word)
        return sync_word

    @sync_word.setter
    def sync_word(self, val: Optional[bytearray]) -> None:
        if val is None:
            self.sync_on = 0
        else:
            assert 1 <= len(val) <= 8
            self._write_from(_REG_SYNC_VALUE1, val)
            self.sync_size = len(val) - 1
            self.sync_on = 1

    @property
    def preamble_length(self) -> int:
        msb = self._read_u8(_REG_PREAMBLE_MSB)
        lsb = self._read_u8(_REG_PREAMBLE_LSB)
        return ((msb << 8) | lsb) & 0xFFFF

    @preamble_length.setter
    def preamble_length(self, val: int) -> None:
        assert 0 <= val <= 65535
        self._write_u8(_REG_PREAMBLE_MSB, (val >> 8) & 0xFF)
        self._write_u8(_REG_PREAMBLE_LSB, val & 0xFF)

    @property
    def frequency_mhz(self) -> float:
        msb = self._read_u8(_REG_FRF_MSB)
        mid = self._read_u8(_REG_FRF_MID)
        lsb = self._read_u8(_REG_FRF_LSB)
        frf = ((msb << 16) | (mid << 8) | lsb) & 0xFFFFFF
        return (frf * _FSTEP) / 1000000.0

    @frequency_mhz.setter
    def frequency_mhz(self, val: float) -> None:
        assert 290 <= val <= 1020
        frf = int((val * 1000000.0) / _FSTEP) & 0xFFFFFF
        self._write_u8(_REG_FRF_MSB, frf >> 16)
        self._write_u8(_REG_FRF_MID, (frf >> 8) & 0xFF)
        self._write_u8(_REG_FRF_LSB, frf & 0xFF)

    @property
    def encryption_key(self) -> Optional[bytearray]:
        if self.aes_on == 0:
            return None
        key = bytearray(16)
        self._read_into(_REG_AES_KEY1, key)
        return key

    @encryption_key.setter
    def encryption_key(self, val: Optional[bytearray]) -> None:
        if val is None:
            self.aes_on = 0
        else:
            assert len(val) == 16
            self._write_from(_REG_AES_KEY1, val)
            self.aes_on = 1

    @property
    def tx_power(self) -> int:
        pa0 = self.pa_0_on
        pa1 = self.pa_1_on
        pa2 = self.pa_2_on
        current_output_power = self.output_power
        if pa0 and not pa1 and not pa2:
            return -18 + current_output_power
        if not pa0 and pa1 and not pa2:
            return -18 + current_output_power
        if not pa0 and pa1 and pa2 and self.high_power and self._tx_power < 18:
            return -14 + current_output_power
        if not pa0 and pa1 and pa2 and self.high_power and self._tx_power >= 18:
            return -11 + current_output_power
        raise RuntimeError("Power amps state unknown!")

    @tx_power.setter
    def tx_power(self, val: float):
        val = int(val)
        pa_0_on = pa_1_on = pa_2_on = 0
        output_power = 0
        if self.high_power:
            assert -2 <= val <= 20
            pa_1_on = 1
            if val <= 13:
                output_power = val + 18
            elif 13 < val <= 17:
                pa_2_on = 1
                output_power = val + 14
            else:
                pa_2_on = 1
                output_power = val + 11
        else:
            assert -18 <= val <= 13
            pa_0_on = 1
            output_power = val + 18
        self.pa_0_on = pa_0_on
        self.pa_1_on = pa_1_on
        self.pa_2_on = pa_2_on
        self.output_power = output_power
        self._tx_power = val

    @property
    def rssi(self) -> float:
        return -self._read_u8(_REG_RSSI_VALUE) / 2.0

    @property
    def bitrate(self) -> float:
        msb = self._read_u8(_REG_BITRATE_MSB)
        lsb = self._read_u8(_REG_BITRATE_LSB)
        return _FXOSC / ((msb << 8) | lsb)

    @bitrate.setter
    def bitrate(self, val: float) -> None:
        assert (_FXOSC / 65535) <= val <= 32000000.0
        bitrate = int((_FXOSC / val) + 0.5) & 0xFFFF
        self._write_u8(_REG_BITRATE_MSB, bitrate >> 8)
        self._write_u8(_REG_BITRATE_LSB, bitrate & 0xFF)

    @property
    def frequency_deviation(self) -> float:
        msb = self._read_u8(_REG_FDEV_MSB)
        lsb = self._read_u8(_REG_FDEV_LSB)
        return _FSTEP * ((msb << 8) | lsb)

    @frequency_deviation.setter
    def frequency_deviation(self, val: float) -> None:
        assert 0 <= val <= (_FSTEP * 16383)
        fdev = int((val / _FSTEP) + 0.5) & 0x3FFF
        self._write_u8(_REG_FDEV_MSB, fdev >> 8)
        self._write_u8(_REG_FDEV_LSB, fdev & 0xFF)

    def packet_sent(self) -> bool:
        return (self._read_u8(_REG_IRQ_FLAGS2) & 0x8) >> 3

    def payload_ready(self) -> bool:
        return (self._read_u8(_REG_IRQ_FLAGS2) & 0x4) >> 2

    def send(
        self,
        data: ReadableBuffer,
        *,
        keep_listening: bool = False,
        destination: Optional[int] = None,
        node: Optional[int] = None,
        identifier: Optional[int] = None,
        flags: Optional[int] = None,
    ) -> bool:
        assert 0 < len(data) <= 60
        self.idle()
        payload = bytearray(5)
        payload[0] = 4 + len(data)
        payload[1] = destination if destination is not None else self.destination
        payload[2] = node if node is not None else self.node
        payload[3] = identifier if identifier is not None else self.identifier
        payload[4] = flags if flags is not None else self.flags
        payload = payload + data  # creates new bytearray; okay for now
        self._write_from(_REG_FIFO, payload)
        self.transmit()
        timed_out = check_timeout(self.packet_sent, self.xmit_timeout)
        if keep_listening:
            self.listen()
        else:
            self.idle()
        return not timed_out

    def send_with_ack(self, data: int) -> bool:
        if self.ack_retries:
            retries_remaining = self.ack_retries
        else:
            retries_remaining = 1
        got_ack = False
        self.sequence_number = (self.sequence_number + 1) & 0xFF
        while not got_ack and retries_remaining:
            self.identifier = self.sequence_number
            self.send(data, keep_listening=True)
            if self.destination == _RH_BROADCAST_ADDRESS:
                got_ack = True
            else:
                ack_packet = self.receive(timeout=self.ack_wait, with_header=True)
                if ack_packet is not None:
                    if ack_packet[3] & _RH_FLAGS_ACK:
                        if ack_packet[2] == self.identifier:
                            got_ack = True
                            break
            if not got_ack:
                time.sleep(self.ack_wait + self.ack_wait * random.random())
            retries_remaining -= 1
            self.flags |= _RH_FLAGS_RETRY
        self.flags = 0
        return got_ack

    def receive(
        self,
        *,
        keep_listening: bool = True,
        with_ack: bool = False,
        timeout: Optional[float] = None,
        with_header: bool = False,
    ):
        timed_out = False
        if timeout is None:
            timeout = self.receive_timeout
        if timeout is not None:
            self.listen()
            timed_out = check_timeout(self.payload_ready, timeout)

        if timed_out and not self.payload_ready():
            if not keep_listening:
                self.idle()
            return None

        self.last_rssi = self.rssi
        self.idle()
        fifo_length = self._read_u8(_REG_FIFO)
        if fifo_length > 0:
            packet = bytearray(fifo_length)
            self._read_into(_REG_FIFO, packet, fifo_length)
        else:
            packet = None

        if fifo_length < 5:
            packet = None
        else:
            if self.node != _RH_BROADCAST_ADDRESS and packet[0] not in {
                _RH_BROADCAST_ADDRESS,
                self.node,
            }:
                packet = None
            elif (
                with_ack
                and ((packet[3] & _RH_FLAGS_ACK) == 0)
                and (packet[0] != _RH_BROADCAST_ADDRESS)
            ):
                if self.ack_delay is not None:
                    time.sleep(self.ack_delay)
                self.send(
                    b"!",
                    destination=packet[1],
                    node=packet[0],
                    identifier=packet[2],
                    flags=(packet[3] | _RH_FLAGS_ACK),
                )
                if (self.seen_ids[packet[1]] == packet[2]) and (packet[3] & _RH_FLAGS_RETRY):
                    packet = None
                else:
                    self.seen_ids[packet[1]] = packet[2]
            if not with_header and packet is not None:
                packet = packet[4:]
        if keep_listening:
            self.listen()
        else:
            self.idle()
        return packet
