# Third-party notices

Newly authored portfolio code is covered by the MIT license in `LICENSE`. Third-party material retains its original license and attribution.

## Kaggle SDK Struct reference

The Struct reference used by the state-copy implementation and its tests is derived from `kaggle_environments/utils.py`.

Copyright 2020 Kaggle Inc. Licensed under the Apache License, Version 2.0. A copy is provided in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt). The upstream project is [Kaggle/kaggle-environments](https://github.com/Kaggle/kaggle-environments).

The reference is a selected class definition for checking state-copy semantics; the complete SDK is not bundled. Extraction and standalone packaging are documented in the code. Reference behavior is preserved, and any portfolio-specific wrapper is identified separately.

## Team-provided competition strategies

The complete R7 agent incorporates team routes and models. Their repositories, modules, and fixed revisions are listed in [docs/CREDITS.md](docs/CREDITS.md). These complete strategies, teachers, model weights, and replay corpora are not redistributed in this repository.
