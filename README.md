This repository contains source code for the SafeTab-P disclosure
avoidance application. SafeTab-P was used by the Census Bureau for the
protection of individual 2020 Census responses in the tabulation and
publication of the Detailed Demographic and Housing Characteristics
File A (Detailed DHC-A). A later source code release will contain the source
code for SafeTab-H, the application used to protect the Detailed
Demographic and Housing Characteristics File B (Detailed DHC-B).

Using the mathematical principles of differential privacy, SafeTab-P infused noise into 2020 Census results to create *privacy-protected statistics* which were used by Census Bureau subject-matter experts to tabulate the 2020 Detailed DHC-A product. SafeTab-P was built on Tumult Analytics, a platform for computing statistics using differential privacy. Both SafeTab-P and the underlying platform are implemented in Python. The latest version of Tumult Analytics can be found at https://tmlt.dev/.

In the interests of both transparency and scientific advancement, the
Census Bureau committed to releasing any source code used in creation
of products protected by formal privacy guarantees. In the case of the 
Detailed Demographic and Housing Characteristics publications, this
includes code developed under contract by Tumult Labs (https://tmlt.io)
and MITRE corporation. Tumult Analytics is an evolving platform and
the code in the repository is from version 0.5.3.

The Census Bureau has already separately released the internally developed
software for the TopDown Algorithm (TDA) used in production of the
2020 Redistricting and the 2020 Demographic and Housing Characteristics
products.

This repository is divided into six subdirectories:
* `configs` contains the specific configuration files used for the
  production Detailed DHC-A runs, including privacy-loss budget (PLB) allocations
  and the rules for adaptive table generation. These configurations reflect
  decisions by the Census Bureau's Data Stewardship Executive Policy committee
  based on experiments conducted by Census Bureau staff.
* `safetab_p` contains the source code for the application itself as used
   to generate the protected statistics used in production.
* `safetab_utils` contains utilities common among the SafeTab products
  developed by Tumult Labs for the Census Bureau.
* `mitre/cef_readers` contains code by MITRE to read the 2020 Census input
  files used by the SafeTab applications.
* `ctools` contains Python utility libraries developed by the Census
  Bureau's DAS team and used by the MITRE CEF readers.
* `tumult` contains the Tumult Analytics platform. This is divided
   into `common`, `analytics`, and `core` directories. The `core` directory
   also includes a pre-packaged Python *wheel* for the core library.

