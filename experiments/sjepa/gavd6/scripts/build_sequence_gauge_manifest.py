#!/usr/bin/env python3

# The implementation is shared with the AMASS-named entry point for backward
# compatibility; its contract accepts any validity-aware Core11 manifest whose
# frame is explicitly gauge-neutral travel or declared unanchored image space.
from build_amass_gauge_manifest import main


if __name__ == "__main__":
    raise SystemExit(main())
