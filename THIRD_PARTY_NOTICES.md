# Third-party material tables

XPOSim code is distributed under the MIT license in [LICENSE](LICENSE).
The bundled tables in `xray_material_data/` are copied from the
[DABAX data repository](https://github.com/oasys-kit/DabaxFiles), which publishes
these files under its MIT license. The upstream copyright and permission notice
are reproduced unchanged in [xray_material_data/LICENSE](xray_material_data/LICENSE).
The XPOSim code license does not replace that notice.

The table bytes, including their original attribution and reference headers,
have not been modified. Sources and checksums are recorded in
[xray_material_data/README.md](xray_material_data/README.md).

The DABAX collection credits the original compilers and scientific sources:

- `AtomicConstants.dat`: the xelem periodic table by Przemek Klosowski (NIST),
  with additional values credited to Kittel in the file header.
- `CrossSec-Compton_McMaster.dat`: McMaster et al., UCRL-50174, and the Brennan
  and Cowan implementation identified in the file header.
- `f1f2_Windt.dat`: David Windt's IMD data collection, combining CXRO and LLNL
  data as described in the file header.

Redistribution here relies on the DABAX repository's MIT grant for its
collection. No claim is made that XPOSim owns these datasets, that all underlying
publications are MIT-licensed, or that the MIT grant licenses IMD software.
Retain the DABAX license and the table headers when redistributing the tables.
Scientific references and data limitations remain relevant independently of
software licensing. No other third-party runtime libraries are bundled.
