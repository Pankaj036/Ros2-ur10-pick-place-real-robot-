#!/usr/bin/env python3
"""
Robotiq FT 300-S  (F-0125)  ROS 2 Driver Node
═══════════════════════════════════════════════
Reads force / torque data from the sensor over Modbus RTU (USB virtual
serial port) and publishes  geometry_msgs/WrenchStamped  on
/ft_sensor/wrench  at ~100 Hz — the same topic the Gazebo plugin uses,
so real and simulated code can share one subscriber.

Wiring / setup
──────────────
  1.  Plug the FT 300-S USB cable into the PC.
  2.  It appears as  /dev/ttyUSB0  (or /dev/ttyUSB1 if something else
      is on ttyUSB0).  Check with:  ls /dev/ttyUSB*
  3.  Add your user to the dialout group if needed:
        sudo usermod -aG dialout $USER   (then log out / in)

Modbus RTU register map  (Robotiq FT 300-S firmware ≥ 2.0)
────────────────────────────────────────────────────────────
  Unit ID   : 0x09  (default; configurable on device)
  Function  : 04  (Read Input Registers)
  Start reg : 0x01F4  (500 decimal)  — Measurement Count
  Registers : 9  (count, status, Fx, Fy, Fz, Tx, Ty, Tz, reserved)

  reg 500  : measurement_count   (uint16, increments each sample)
  reg 501  : status              (uint16, 0 = OK)
  reg 502  : Fx * 100            (int16  → divide by 100 for N)
  reg 503  : Fy * 100            (int16)
  reg 504  : Fz * 100            (int16)
  reg 505  : Tx * 10000          (int16  → divide by 10000 for N·m)
  reg 506  : Ty * 10000          (int16)
  reg 507  : Tz * 10000          (int16)

  Verify these addresses against your firmware version's user manual.
  If you have an older device the start address may be 0x0050 (80).

Usage
─────
  # standalone
  python3 ft300_driver_node.py --port /dev/ttyUSB0

  # via launch file (port overridable with ROS param)
  ros2 run ur_yt_sim ft300_driver_node --ros-args -p port:=/dev/ttyUSB0
"""

import argparse
import struct
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header

try:
    from pymodbus.client import ModbusSerialClient
    PYMODBUS_V3 = True
except ImportError:
    from pymodbus.client.sync import ModbusSerialClient
    PYMODBUS_V3 = False


# ── Sensor Modbus constants ────────────────────────────────────────────────────
_UNIT_ID        = 0x09       # default slave / unit ID
_REG_START      = 500        # 0x01F4 — first input register
_REG_COUNT      = 8          # count, status, Fx, Fy, Fz, Tx, Ty, Tz
_FORCE_SCALE    = 100.0      # raw int → N
_TORQUE_SCALE   = 10000.0    # raw int → N·m
_DEFAULT_PORT   = "/dev/ttyUSB0"
_BAUD            = 19200
_RATE_HZ        = 100


def _to_signed16(val: int) -> int:
    """Convert an unsigned 16-bit Modbus register value to signed int16."""
    return struct.unpack("h", struct.pack("H", val & 0xFFFF))[0]


class FT300DriverNode(Node):
    def __init__(self):
        super().__init__("ft300_driver")

        self.declare_parameter("port",      _DEFAULT_PORT)
        self.declare_parameter("baud",      _BAUD)
        self.declare_parameter("unit_id",   _UNIT_ID)
        self.declare_parameter("frame_id",  "ft300_sensor_frame")
        self.declare_parameter("rate_hz",   float(_RATE_HZ))

        port     = self.get_parameter("port").value
        baud     = self.get_parameter("baud").value
        unit_id  = self.get_parameter("unit_id").value
        frame_id = self.get_parameter("frame_id").value
        rate_hz  = self.get_parameter("rate_hz").value

        self._unit_id  = unit_id
        self._frame_id = frame_id

        self._pub = self.create_publisher(WrenchStamped, "/ft_sensor/wrench", 10)

        # ── Connect Modbus client ──────────────────────────────────────────
        if PYMODBUS_V3:
            self._client = ModbusSerialClient(
                port=port,
                baudrate=baud,
                bytesize=8,
                parity="N",
                stopbits=1,
                timeout=1,
            )
        else:
            self._client = ModbusSerialClient(
                method="rtu",
                port=port,
                baudrate=baud,
                bytesize=8,
                parity="N",
                stopbits=1,
                timeout=1,
            )

        if not self._client.connect():
            self.get_logger().error(
                f"Cannot open serial port {port}. "
                "Check cable, permissions (dialout group), and --port argument."
            )
            raise RuntimeError(f"Failed to connect to FT300 on {port}")

        self.get_logger().info(
            f"FT 300-S connected on {port} @ {baud} baud  "
            f"unit_id={unit_id}  publishing → /ft_sensor/wrench"
        )

        period = 1.0 / rate_hz
        self.create_timer(period, self._read_and_publish)

    def _read_and_publish(self):
        try:
            result = self._client.read_input_registers(
                address=_REG_START,
                count=_REG_COUNT,
                slave=self._unit_id,
            )
        except Exception as exc:
            self.get_logger().warn(f"Modbus read error: {exc}", throttle_duration_sec=5.0)
            return

        if result is None or result.isError():
            self.get_logger().warn("FT300 Modbus read returned error", throttle_duration_sec=5.0)
            return

        regs = result.registers
        if len(regs) < 8:
            self.get_logger().warn(f"Short register read: got {len(regs)} regs", throttle_duration_sec=5.0)
            return

        # regs[0] = measurement_count, regs[1] = status
        fx = _to_signed16(regs[2]) / _FORCE_SCALE
        fy = _to_signed16(regs[3]) / _FORCE_SCALE
        fz = _to_signed16(regs[4]) / _FORCE_SCALE
        tx = _to_signed16(regs[5]) / _TORQUE_SCALE
        ty = _to_signed16(regs[6]) / _TORQUE_SCALE
        tz = _to_signed16(regs[7]) / _TORQUE_SCALE

        msg = WrenchStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.wrench.force.x  = fx
        msg.wrench.force.y  = fy
        msg.wrench.force.z  = fz
        msg.wrench.torque.x = tx
        msg.wrench.torque.y = ty
        msg.wrench.torque.z = tz
        self._pub.publish(msg)

    def destroy_node(self):
        self._client.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = FT300DriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
