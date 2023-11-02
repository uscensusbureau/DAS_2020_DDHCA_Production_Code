"""The schemas for SafeTab input files."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Tumult Labs
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

GEO_SCHEMA = {
    "TABBLKST": "VARCHAR",  # Note: spec says `Number` but has leading zero
    "TABBLKCOU": "VARCHAR",
    "TABTRACTCE": "VARCHAR",
    "TABBLK": "VARCHAR",
    "PLACEFP": "VARCHAR",
    "AIANNHCE": "VARCHAR",
}
"""SafeTab schema for the geo df."""

UNIT_SCHEMA = {
    "TABBLKST": "VARCHAR",  # Note: spec says `Number` but has leading zero
    "TABBLKCOU": "VARCHAR",
    "TABTRACTCE": "VARCHAR",
    "TABBLK": "VARCHAR",
    "HHRACE": "VARCHAR",
    "QRACE1": "VARCHAR",
    "QRACE2": "VARCHAR",
    "QRACE3": "VARCHAR",
    "QRACE4": "VARCHAR",
    "QRACE5": "VARCHAR",
    "QRACE6": "VARCHAR",
    "QRACE7": "VARCHAR",
    "QRACE8": "VARCHAR",
    "QSPAN": "VARCHAR",
    "HOUSEHOLD_TYPE": "INTEGER",
    "TEN": "INTEGER",
}
"""SafeTab schema for the unit df."""

PERSON_SCHEMA = {
    "QAGE": "INTEGER",
    "QSEX": "VARCHAR",
    "HOUSEHOLDER": "VARCHAR",
    "TABBLKST": "VARCHAR",  # Note: spec says `Number` but has leading zero
    "TABBLKCOU": "VARCHAR",
    "TABTRACTCE": "VARCHAR",
    "TABBLK": "VARCHAR",
    "CENRACE": "VARCHAR",
    "QRACE1": "VARCHAR",  # Note: spec says `Number` but should be Varchar
    "QRACE2": "VARCHAR",
    "QRACE3": "VARCHAR",
    "QRACE4": "VARCHAR",
    "QRACE5": "VARCHAR",
    "QRACE6": "VARCHAR",
    "QRACE7": "VARCHAR",
    "QRACE8": "VARCHAR",
    "QSPAN": "VARCHAR",
}
"""SafeTab schema for the person df."""

POP_GROUP_TOTAL_SCHEMA = {
    "REGION_ID": "VARCHAR",
    "REGION_TYPE": "VARCHAR",
    "ITERATION_CODE": "VARCHAR",
    "COUNT": "INTEGER",
}
