This repository contains source code for the SafeTab-P disclosure
avoidance application. SafeTab-P was used by the Census Bureau for the
protection of individual 2020 Census responses in the tabulation and
publication of the Detailed Demographic and Housing Characteristics
File A (DDHC-A). A later source code release will contain the source
code for SafeTab-H, the application used to protect the Detailed
Demographic and Housing Characteristics File B (DDHC-B).

Using the mathematical principles of formal privacy, SafeTab-P infused
noise into Census survey results to create *privacy-protected
microdata* which were used by Bureau subject matter experts to
tabulate the 2020 DDHC-A product.  SafeTab-P was built on Tumult's
"Analytics" and "Core" platforms. both SafeTab-P and the underlying
platforms are implemented in Python. The latest version of the
platforms can be found at https://tmlt.dev/.

In the interests of both transparency and scientific advancement, the
Census Bureau committed to releasing any source code used in creation
of products protected by formal privacy guarantees. In the case of the
the Detailed Demographic & Housing Characteristics publications, this
includes code developed under contract by Tumult Software (tmlt.io)
and MITRE corporation. Tumult's underlying platform is evolving and
the code in the repository is a snapshot of the code used for the
DDHC-A.

The bureau has already separately released the internally developed
software for the Top Down Algorithm (TDA) used in production of the
2020 Redistricting and the 2020 Demographic & Housing Characteristics
products.

This software for this repository is divided into five subdirectories:
* `configs` contains the specific configuration files used for the
  production DDHC-A runs, including privacy loss budget (PLB) allocations
  and the rules for adaptive table generation. These configurations reflect
  decisions by the Bureau's DSEP (Data Stewardship Executive Policy) committee
  based on experiments conducted by Census Bureau staff.
* `safetab_p` contains the source code for the application itself as used
   to generate the protected microdata used in production.
* `safetab_utils` contains utilities common among the SafeTab products
  developed by Tumult for the Census Bureau.
* `mitre/cef_readers` contains code by MITRE to read the Census input
  files used by the SafeTab applications.
* `ctools` contains Python utility libraries developed the the Census
  Bureau's DAS team and used by the MITRE CEF readers.
* `tumult` contains the Tumult Analytics platform. This is divided
   into `common`, `analytics`, and `core` directories. The `core` directory
   also includes a pre-packaged Python *wheel* for the core library.

