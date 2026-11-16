# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Griffe extension to document functions bound to the ``Cutout`` class.

``Cutout`` exposes module-level functions as methods, e.g. ``wind = wind``.
Static analysis sees these as attributes without docstrings. This extension
replaces them with aliases of the functions, so that they are documented with
their signatures and docstrings.
"""

from __future__ import annotations

from typing import Any

import griffe


class CutoutMethods(griffe.Extension):
    """Replace class attributes that bind imported functions with aliases."""

    def on_class_members(self, *, cls: griffe.Class, **kwargs: Any) -> None:
        """Alias attributes of ``atlite.cutout.Cutout`` assigned from imports."""
        if cls.path != "atlite.cutout.Cutout":
            return
        imports = cls.module.imports
        for name, member in list(cls.members.items()):
            if member.is_attribute and str(member.value) in imports:
                target = imports[str(member.value)]
                cls.set_member(name, griffe.Alias(name, target, parent=cls))
