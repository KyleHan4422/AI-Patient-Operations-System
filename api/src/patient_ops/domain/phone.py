"""Phone numbers in one canonical form (E.164), so that equal means the same line.

"(212) 555-0143", "212.555.0143" and "+1 212 555 0143" are one patient. Stored
as typed, the UNIQUE constraint on patients.phone would see three.

Deliberately small: North American numbers plus explicit international "+"
numbers. A production system would use libphonenumber (the `phonenumbers`
package) for per-country rules.
"""

from __future__ import annotations

import re

_NANP_COUNTRY_CODE = "1"


def normalize_phone(raw: str) -> str:
    """Return the E.164 form ("+12125550143"). Raises ValueError if unrecognised."""
    digits = re.sub(r"\D", "", raw)
    if raw.strip().startswith("+"):
        if 8 <= len(digits) <= 15:  # E.164 allows at most 15 digits
            return f"+{digits}"
    elif len(digits) == 10:
        return f"+{_NANP_COUNTRY_CODE}{digits}"
    elif len(digits) == 11 and digits.startswith(_NANP_COUNTRY_CODE):
        return f"+{digits}"
    raise ValueError(f"unrecognised phone number: {raw!r}")
