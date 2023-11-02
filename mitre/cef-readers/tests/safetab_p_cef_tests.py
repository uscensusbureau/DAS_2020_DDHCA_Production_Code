from phsafe_safetab_reader.safetab_p_cef_reader import CEFPReader
from phsafe_safetab_reader.safetab_h_cef_reader import CEFHReader

def test_pr_only():
    reader = CEFPReader("src/phsafe_safetab_reader/safetab_cef_config_2010.ini", ['72'])
    person_df = reader.test_per_df
    unit_df = reader.test_unit_df
    geo_df = reader.test_geo_df

    per_states = [row.bcustatefp for row in person_df.select('bcustatefp').distinct().collect()]
    assert set(per_states) == set(['72'])

    unit_states = [row.bcustatefp for row in unit_df.select('bcustatefp').distinct().collect()]
    assert set(unit_states) == set(['72'])

    geo_states = [row.TABBLKST for row in geo_df.select('TABBLKST').distinct().collect()]
    assert set(geo_states) == set(['72'])

    print('puerto rico input test passed')

    print('united states input test passed')

def test_us_only():
    reader = CEFPReader("src/phsafe_safetab_reader/safetab_cef_config_2010.ini", ['01', '02'])
    person_df = reader.test_per_df
    unit_df = reader.test_unit_df
    geo_df = reader.test_geo_df

    per_states = [row.bcustatefp for row in person_df.select('bcustatefp').distinct().collect()]
    assert set(per_states) == set(['01', '02'])

    unit_states = [row.bcustatefp for row in unit_df.select('bcustatefp').distinct().collect()]
    assert set(unit_states) == set(['01', '02'])

    geo_states = [row.TABBLKST for row in geo_df.select('TABBLKST').distinct().collect()]
    assert set(geo_states) == set(['01', '02'])

    print('united states input test passed')

def test_pr_and_us():
    reader = CEFPReader("src/phsafe_safetab_reader/safetab_cef_config_2010.ini", ['72', '01', '02'])
    person_df = reader.test_per_df
    unit_df = reader.test_unit_df
    geo_df = reader.test_geo_df

    per_states = [row.bcustatefp for row in person_df.select('bcustatefp').distinct().collect()]
    assert set(per_states) == set(['72', '01', '02'])

    unit_states = [row.bcustatefp for row in unit_df.select('bcustatefp').distinct().collect()]
    assert set(unit_states) == set(['72', '01', '02'])

    geo_states = [row.TABBLKST for row in geo_df.select('TABBLKST').distinct().collect()]
    assert set(geo_states) == set(['72', '01', '02'])

    print('puerto rico and united states input test passed')

def test_allstates_and_pr():
    reader = CEFPReader("src/phsafe_safetab_reader/safetab_cef_config_2010.ini", [])
    person_df = reader.test_per_df
    unit_df = reader.test_unit_df
    geo_df = reader.test_geo_df

    all_state_fips = set(['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56', '72'])

    per_states = [row.bcustatefp for row in person_df.select('bcustatefp').distinct().collect()]
    assert set(per_states) == all_state_fips

    unit_states = [row.bcustatefp for row in unit_df.select('bcustatefp').distinct().collect()]
    assert set(unit_states) == all_state_fips

    geo_states = [row.TABBLKST for row in geo_df.select('TABBLKST').distinct().collect()]
    assert set(geo_states) == all_state_fips

    print('puerto rico and all united states input test passed')

def test_input_types():
    reader = CEFPReader("src/phsafe_safetab_reader/safetab_cef_config_2010.ini", ['72', '01', '02'])
    person_df = reader.test_per_df
    unit_df = reader.test_unit_df
    geo_df = reader.test_geo_df

    per_types = [row[1] for row in person_df.dtypes]
    assert per_types == ['string'] + 2 * ['bigint'] + 2 * ['string', 'bigint'] + 5 * ['bigint'] + ['string'] + 2 * ['bigint'] + 13 * ['string']

    unit_types = [row[1] for row in unit_df.dtypes]
    assert unit_types == 2 * ['string', 'bigint'] + 11 * ['string'] + ['bigint'] + 6 * ['string'] + 4 * ['bigint'] + 2 * ['string']

    geo_fields = ['TABBLKST', 'TABBLKCOU', 'TABTRACTCE', 'TABBLK', 'PLACEFP', 'AIANNHCE', 'OIDTABBLK']
    geo_types = [row[1] for row in geo_df.dtypes if row[0] in geo_fields]
    print('geotypes', geo_types)
    assert geo_types == 6 * ['string'] + ['bigint']

    print('input types test passed')

def test_output_types_and_order():
    reader = CEFPReader("src/phsafe_safetab_reader/safetab_cef_config_2010.ini", ['72', '01', '02'])

    person_df = reader.get_person_df()
    # unit_df = reader.get_unit_df()
    geo_df = reader.get_geo_df()

    per_types = [row for row in person_df.dtypes]
    assert per_types == [('QAGE', 'bigint'), ('QSEX', 'string'), ('HOUSEHOLDER', 'string'), ('TABBLKST', 'string'), ('TABBLKCOU', 'string'), ('TABTRACTCE', 'string'), ('TABBLK', 'string'), ('CENRACE', 'string'), ('QRACE1', 'string'), ('QRACE2', 'string'), ('QRACE3', 'string'), ('QRACE4', 'string'), ('QRACE5', 'string'), ('QRACE6', 'string'), ('QRACE7', 'string'), ('QRACE8', 'string'), ('QSPAN', 'string')]

    # unit_types = [row for row in unit_df.dtypes]
    # assert unit_types == [('TABBLKST', 'string'), ('TABBLKCOU', 'string'), ('TABTRACTCE', 'string'), ('TABBLK', 'string'), ('HHRACE', 'string'), ('QRACE1', 'string'), ('QRACE2', 'string'), ('QRACE3', 'string'), ('QRACE4', 'string'), ('QRACE5', 'string'), ('QRACE6', 'string'), ('QRACE7', 'string'), ('QRACE8', 'string'), ('QSPAN', 'string'), ('HOUSEHOLD_TYPE', 'int'), ('TEN', 'int')]

    geo_fields = ['TABBLKST', 'TABBLKCOU', 'TABTRACTCE', 'TABBLK', 'PLACEFP', 'AIANNHCE']
    geo_types = [row[1] for row in geo_df.dtypes if row[0] in geo_fields]
    assert geo_types == 6 * ['string']

    print('output types and order test passed')

def run_all_tests():
    test_pr_only()
    test_us_only()
    test_pr_and_us()
    test_allstates_and_pr()
    test_input_types()
    test_output_types_and_order()

run_all_tests()
