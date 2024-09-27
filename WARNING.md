# Warning: `pyarrow` library vulnerability

There is a known vulnerability in the binary Python wheel at:
    `tumult/core/tmlt_core-0.6.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`

This wheel contains a binary version of the `pyarrow` module with a
vulnerability identified in 2023
(https://nvd.nist.gov/vuln/detail/CVE-2023-47248). The SAFETAB-P code
itself does not expose this vulnerability, but any modified or
extended versions should rebuild the wheel from sources beneath the
`tumult/core` prefix or later sources from Tumult itself.


