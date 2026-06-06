from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any


URDF_ACTION_SCHEMA_VERSION = "urdf_action_schema.v1"
URDF_ACTION_SCHEMA_SOURCE = "urdf"
URDF_ACTION_UNITS_NATIVE = "urdf-native"

_ACTUATED_JOINT_TYPES = frozenset({"continuous", "prismatic", "revolute"})


@dataclass(frozen=True, slots=True)
class UrdfActionJoint:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    axis: tuple[float, float, float]
    lower: float | None
    upper: float | None
    velocity: float | None
    effort: float | None
    units: str


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_finite_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_axis(value: str | None) -> tuple[float, float, float]:
    if not value:
        return (1.0, 0.0, 0.0)
    parts = value.split()
    if len(parts) != 3:
        return (1.0, 0.0, 0.0)
    parsed = tuple(_parse_finite_float(part) for part in parts)
    if any(part is None for part in parsed):
        return (1.0, 0.0, 0.0)
    return parsed  # type: ignore[return-value]


def _child_attr(element: ET.Element, child_tag: str, attr_name: str) -> str:
    child = next((candidate for candidate in element if _local_tag(candidate.tag) == child_tag), None)
    if child is None:
        return ""
    return (child.attrib.get(attr_name) or "").strip()


def _joint_units(joint_type: str) -> str:
    return "meters" if joint_type == "prismatic" else "radians"


def _build_joint(element: ET.Element) -> UrdfActionJoint | None:
    joint_type = (element.attrib.get("type") or "").strip()
    if joint_type not in _ACTUATED_JOINT_TYPES:
        return None

    name = (element.attrib.get("name") or "").strip()
    parent_link = _child_attr(element, "parent", "link")
    child_link = _child_attr(element, "child", "link")
    if not name or not parent_link or not child_link:
        return None

    axis_element = next((candidate for candidate in element if _local_tag(candidate.tag) == "axis"), None)
    limit_element = next((candidate for candidate in element if _local_tag(candidate.tag) == "limit"), None)
    lower = None
    upper = None
    velocity = None
    effort = None
    if limit_element is not None:
        lower = _parse_finite_float(limit_element.attrib.get("lower"))
        upper = _parse_finite_float(limit_element.attrib.get("upper"))
        velocity = _parse_finite_float(limit_element.attrib.get("velocity"))
        effort = _parse_finite_float(limit_element.attrib.get("effort"))

    if joint_type == "continuous":
        lower = None
        upper = None

    return UrdfActionJoint(
        name=name,
        joint_type=joint_type,
        parent_link=parent_link,
        child_link=child_link,
        axis=_parse_axis(axis_element.attrib.get("xyz") if axis_element is not None else None),
        lower=lower,
        upper=upper,
        velocity=velocity,
        effort=effort,
        units=_joint_units(joint_type),
    )


def _root_links(links: list[str], joints: list[UrdfActionJoint]) -> list[str]:
    children = {joint.child_link for joint in joints}
    roots = [link for link in links if link not in children]
    return roots or ([joints[0].parent_link] if joints else [])


def _order_joints_by_tree(links: list[str], joints: list[UrdfActionJoint]) -> list[UrdfActionJoint]:
    by_parent: dict[str, list[UrdfActionJoint]] = {}
    for joint in joints:
        by_parent.setdefault(joint.parent_link, []).append(joint)

    ordered: list[UrdfActionJoint] = []
    visited: set[str] = set()

    def walk(link_name: str) -> None:
        for joint in by_parent.get(link_name, []):
            if joint.name not in visited:
                ordered.append(joint)
                visited.add(joint.name)
            walk(joint.child_link)

    for link in _root_links(links, joints):
        walk(link)

    for joint in joints:
        if joint.name not in visited:
            ordered.append(joint)
    return ordered


def _joint_payload(joint: UrdfActionJoint) -> dict[str, Any]:
    return {
        "name": joint.name,
        "type": joint.joint_type,
        "parent_link": joint.parent_link,
        "child_link": joint.child_link,
        "axis": list(joint.axis),
        "lower": joint.lower,
        "upper": joint.upper,
        "velocity": joint.velocity,
        "effort": joint.effort,
        "units": joint.units,
    }


def build_urdf_action_schema(
    urdf_xml: str,
    *,
    robot_name: str | None = None,
    action_units: str = URDF_ACTION_UNITS_NATIVE,
) -> dict[str, Any]:
    """Build a deterministic action schema from any URDF's actuated joints."""

    if not urdf_xml.strip():
        raise ValueError("DreamZero URDF action schema requires non-empty URDF content.")

    try:
        root = ET.fromstring(urdf_xml)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse URDF XML: {exc}") from exc

    if _local_tag(root.tag) != "robot":
        raise ValueError("URDF action schema requires a <robot> root element.")

    links = [
        (element.attrib.get("name") or "").strip()
        for element in root
        if _local_tag(element.tag) == "link"
    ]
    links = [link for link in links if link]

    joints = [
        joint
        for element in root
        if _local_tag(element.tag) == "joint"
        for joint in [_build_joint(element)]
        if joint is not None
    ]
    ordered_joints = _order_joints_by_tree(links, joints)
    if not ordered_joints:
        raise ValueError("URDF action schema requires at least one actuated joint.")

    resolved_robot_name = (robot_name or root.attrib.get("name") or "robot").strip() or "robot"
    joint_names = [joint.name for joint in ordered_joints]

    return {
        "schema_version": URDF_ACTION_SCHEMA_VERSION,
        "source": URDF_ACTION_SCHEMA_SOURCE,
        "robot_name": resolved_robot_name,
        "urdf_hash": hashlib.sha256(urdf_xml.encode("utf-8")).hexdigest(),
        "action_dim": len(joint_names),
        "joint_names": joint_names,
        "action_units": action_units,
        "joints": [_joint_payload(joint) for joint in ordered_joints],
    }
