"""Public constants for the profile-evaluation black box."""

PROTOCOL_VERSION = "profile-evaluation-1.0"
STAGE_PROTOCOL_VERSION = "fact-block-stages-1.0"
PUBLIC_TASKS = ("completeness", "support", "reliability")
PUBLIC_LABELS = {
    "completeness": ("C", "P", "M", "D", "U"),
    "support": ("S", "X", "A", "G", "V", "U", "C"),
    "reliability": ("clean", "error"),
}
