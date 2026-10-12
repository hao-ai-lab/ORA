"""Native import surface for shared public RGB observation projections."""

from ora.agent.context.observation_projection import (
    AUXILIARY_CONTACT_SHEET_CAMERAS,
    AUXILIARY_CONTACT_SHEET_LAYOUTS,
    AUXILIARY_CONTACT_SHEET_SCHEMA,
    AUXILIARY_CONTACT_SHEET_VIEW,
    ObservationProjectionError,
    materialize_auxiliary_contact_sheet,
)

__all__ = [
    "AUXILIARY_CONTACT_SHEET_CAMERAS", "AUXILIARY_CONTACT_SHEET_LAYOUTS",
    "AUXILIARY_CONTACT_SHEET_SCHEMA",
    "AUXILIARY_CONTACT_SHEET_VIEW", "ObservationProjectionError",
    "materialize_auxiliary_contact_sheet",
]
