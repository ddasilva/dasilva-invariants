#!/bin/env python
"""Update geopack database"""

import os
import PyGeopack as gp


def main():
    # Check environmental variables. Allow running without them set, but issue
    # warning
    required_vars = ["KPDATA_PATH", "OMNIDATA_PATH", "GEOPACK_PATH"]

    for var in required_vars:
        if var in os.environ:
            print(f"{var} = {repr(os.environ[var])}")
        else:
            print(f"Environment variable {var} missing!")

    # Run UpdateParameters() method
    gp.Params.UpdateParameters(SkipWParameters=False)


if __name__ == "__main__":
    main()
