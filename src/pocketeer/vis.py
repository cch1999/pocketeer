"""Visualization utilities for pocketeer using atomworks."""

import colorsys
import contextlib
from typing import Any

import biotite.structure as struc  # type: ignore

# Suppress Atomworks import messages about env variables
with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    from atomworks.io.utils.visualize import view  # type: ignore[import-untyped]

from pocketeer.core.types import Pocket


def view_pockets(
    atomarray: struc.AtomArray,
    pockets: list[Pocket],
    color_scheme: str = "rainbow",
    sphere_opacity: float = 0.7,
    sphere_scale: float = 1.0,
    receptor_cartoon: bool = True,
    receptor_surface: bool = False,
    **kwargs: Any,
) -> Any:
    """Visualize protein structure with detected pockets using atomworks.

    Note: This function largely just a wrapper around atomworks.io.utils.visualize.view.
    For more advanced visualization, use the kwargs to pass to view and consults the docs:
    https://baker-laboratory.github.io/atomworks-dev/latest/io/utils/visualize.html

    Args:
        atomarray: Biotite AtomArray with structure data
        pockets: List of Pocket objects from pocketeer.find_pockets()
        color_scheme: Color scheme for pockets ("rainbow", "grayscale", "red_blue")
        sphere_opacity: Opacity of pocket spheres (0.0 to 1.0)
        sphere_scale: Scale factor for sphere sizes
        receptor_style: Visualization style for protein ("cartoon", "stick", "sphere")
        receptor_surface: Show surface for the protein structure
        **kwargs: Additional keyword arguments for the viewer

    Returns:
        atomworks viewer instance

    Raises:
        ImportError: if atomworks is not installed
        TypeError: if atomarray is not a Biotite AtomArray

    Examples:
        >>> import pocketeer
        >>> atoms = pocketeer.load_structure("protein.pdb")
        >>> pockets = pocketeer.find_pockets(atoms)
        >>> viewer = pocketeer.view_pockets(atoms, pockets)
        >>> viewer.show()  # In Jupyter notebook
    """

    if not pockets:
        raise ValueError("No pockets found")

    # Validate input
    if not isinstance(atomarray, struc.AtomArray):
        raise TypeError(
            f"atomarray must be a Biotite AtomArray. Got {type(atomarray).__name__}. "
            "Use pocketeer.load_structure() to load from PDB file."
        )

    # Create viewer and add structure
    # See https://baker-laboratory.github.io/atomworks-dev/latest/io/utils/visualize.html for more details # noqa: E501
    viewer = view(atomarray, show_cartoon=receptor_cartoon, show_surface=receptor_surface, **kwargs)

    # Generate pocket colors
    pocket_ids = list(set(pocket.pocket_id for pocket in pockets))
    pocket_colors: dict[int, str] = {}

    if color_scheme == "rainbow":
        for i, pocket_id in enumerate(pocket_ids):
            hue = i / len(pocket_ids)
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            pocket_colors[pocket_id] = (
                f"rgb({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)})"
            )
    elif color_scheme == "grayscale":
        for i, pocket_id in enumerate(pocket_ids):
            gray = int(128 + (i * 127) / len(pocket_ids))
            pocket_colors[pocket_id] = f"rgb({gray},{gray},{gray})"
    elif color_scheme == "red_blue":
        for i, pocket_id in enumerate(pocket_ids):
            pocket_colors[pocket_id] = "rgb(255,100,100)" if i % 2 == 0 else "rgb(100,100,255)"
    else:
        raise ValueError(
            f"Unknown color scheme: {color_scheme}. Choose from: 'rainbow', 'grayscale', 'red_blue'"
        )

    # Add pocket spheres to viewer
    for pocket in pockets:
        color = pocket_colors.get(pocket.pocket_id, "rgb(128,128,128)")

        for sphere in pocket.spheres:
            center = (
                sphere.center.tolist() if hasattr(sphere.center, "tolist") else list(sphere.center)
            )

            # Add sphere using py3Dmol API
            viewer.addSphere(
                {
                    "center": {"x": center[0], "y": center[1], "z": center[2]},
                    "radius": float(sphere.radius) * sphere_scale,
                    "color": color,
                    "opacity": sphere_opacity,
                }
            )

    # Zoom to fit all content
    viewer.zoomTo()

    return viewer
