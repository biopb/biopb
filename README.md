# napari-biopb

[![License MIT](https://img.shields.io/pypi/l/napari-biopb.svg?color=green)](https://github.com/jiyuuchc/napari-biopb/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-biopb.svg?color=green)](https://pypi.org/project/napari-biopb)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-biopb.svg?color=green)](https://python.org)
[![tests](https://github.com/jiyuuchc/napari-biopb/workflows/tests/badge.svg)](https://github.com/jiyuuchc/napari-biopb/actions)
[![codecov](https://codecov.io/gh/jiyuuchc/napari-biopb/branch/main/graph/badge.svg)](https://codecov.io/gh/jiyuuchc/napari-biopb)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-biopb)](https://napari-hub.org/plugins/napari-biopb)

A [biopb](https://github.com/jiyuuchc/biopb) plugin for [napari](https://github.com/napari/napari).

## Installation

### Bundled Installers

Pre-built standalone installers are available from the [Releases](https://github.com/jiyuuchc/napari-biopb/releases) page. These include napari with the napari-biopb plugin pre-installed—no Python setup required.

| Platform | Download |
|----------|----------|
| Windows | `napari-biopb-windows.zip` |
| macOS (Intel) | `napari-biopb-macos-intel.tar.gz` |
| macOS (Apple Silicon) | `napari-biopb-macos-arm.tar.gz` |
| Linux | `napari-biopb-linux.tar.gz` |

**Usage:**
- **Windows**: Extract the zip and run `napari-biopb.exe`
- **macOS**: Extract and open `napari-biopb.app` (right-click → "Open" to bypass Gatekeeper for unsigned apps)
- **Linux**: Extract and run `./napari-biopb`

### From PyPI

For users with existing virtual environement setup for napari: install `napari-biopb` via [pip]:

    pip install napari-biopb

### Development Version

To install latest development version:

    pip install git+https://github.com/jiyuuchc/napari-biopb.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-biopb" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/jiyuuchc/napari-biopb/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
