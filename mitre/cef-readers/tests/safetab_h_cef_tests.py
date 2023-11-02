from phsafe_safetab_reader.safetab_h_cef_reader import CEFHReader
from pyspark.sql.functions import col, substring

def test_us_only():
    reader = CEFHReader("src/phsafe_safetab_reader/safetab_h_cef_config_2010.ini", ['01', '02'])
    pop_df = reader.get_pop_group_details_df()
    unit_df = reader.get_unit_df()
    geo_df = reader.get_geo_df()

    unit_states = [row.TABBLKST for row in unit_df.select('TABBLKST').distinct().collect()]
    assert set(unit_states) == set(['01', '02'])

    geo_states = [row.TABBLKST for row in geo_df.select('TABBLKST').distinct().collect()]
    assert set(geo_states) == set(['01', '02'])

    assert pop_df.schema["COUNT"].dataType.typeName() == "integer"

    print('united states input test passed')


def test_pr_only():
    reader = CEFHReader("src/phsafe_safetab_reader/safetab_h_cef_config_2010.ini", ['72'])
    unit_df = reader.get_unit_df()
    geo_df = reader.get_geo_df()

    #per_states = [row.TABBLKST for row in person_df.select('TABBLKST').distinct().collect()]
    #assert set(per_states) == set(['72'])

    unit_states = [row.TABBLKST for row in unit_df.select('TABBLKST').distinct().collect()]
    assert set(unit_states) == set(['72'])

    geo_states = [row.TABBLKST for row in geo_df.select('TABBLKST').distinct().collect()]
    assert set(geo_states) == set(['72'])

    print('puerto rico input test passed')

def test_pr_and_us():
    reader = CEFHReader("src/phsafe_safetab_reader/safetab_h_cef_config_2010.ini", ['72', '01', '02'])
    unit_df = reader.get_unit_df()
    geo_df = reader.get_geo_df()

    unit_states = [row.TABBLKST for row in unit_df.select('TABBLKST').distinct().collect()]
    assert set(unit_states) == set(['72', '01', '02'])

    geo_states = [row.TABBLKST for row in geo_df.select('TABBLKST').distinct().collect()]
    assert set(geo_states) == set(['72', '01', '02'])

    print('puerto rico and united states input test passed')

def test_allstates_and_pr():
    reader = CEFHReader("src/phsafe_safetab_reader/safetab_h_cef_config_2010.ini", [])
    unit_df = reader.get_unit_df()
    geo_df = reader.get_geo_df()

    all_state_fips = set(['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56', '72'])

    unit_states = [row.TABBLKST for row in unit_df.select('TABBLKST').distinct().collect()]
    assert set(unit_states) == all_state_fips

    geo_states = [row.TABBLKST for row in geo_df.select('TABBLKST').distinct().collect()]
    assert set(geo_states) == all_state_fips

    print('puerto rico and all united states input test passed')

def test_input_types():
    reader = CEFHReader("src/phsafe_safetab_reader/safetab_h_cef_config_2010.ini", ['72', '01', '02'])
    unit_df = reader.get_unit_df_input()
    geo_df = reader.get_geo_df_input()
    pop_df = reader.get_pop_df_input()

    unit_types = [row[1] for row in unit_df.dtypes]
    assert unit_types == 2 * ['string', 'bigint'] + 11 * ['string'] + ['bigint'] + 6 * ['string'] + 4 * ['bigint'] + 2 * ['string']

    geo_fields = ['TABBLKST', 'TABBLKCOU', 'TABTRACTCE', 'TABBLK', 'PLACEFP', 'AIANNHCE', 'OIDTABBLK']
    geo_types = [row[1] for row in geo_df.dtypes if row[0] in geo_fields]
    print('geotypes', geo_types)
    assert geo_types == 6 * ['string'] + ['bigint']

    pop_types = [row[1] for row in pop_df.dtypes]
    print(pop_types)
    assert pop_types == 4 * ['string']

    print('input types test passed')

    #print('pop_df.count rows: ', pop_df.shape[0])
    #print('pop_df.count cols: ', pop_df.shape[1])
    print('pop_df.count() : : ', pop_df.count())


def test_output_types_and_order():
    reader = CEFHReader("src/phsafe_safetab_reader/safetab_h_cef_config_2010.ini", ['72', '01', '02'])

    unit_df = reader.get_unit_df()
    geo_df = reader.get_geo_df()
    pop_df = reader.get_pop_group_details_df()

    unit_types = [row for row in unit_df.dtypes]
    assert unit_types == [('TABBLKST', 'string'), ('TABBLKCOU', 'string'), ('TABTRACTCE', 'string'), ('TABBLK', 'string'), ('HHRACE', 'string'), ('QRACE1', 'string'), ('QRACE2', 'string'), ('QRACE3', 'string'), ('QRACE4', 'string'), ('QRACE5', 'string'), ('QRACE6', 'string'), ('QRACE7', 'string'), ('QRACE8', 'string'), ('QSPAN', 'string'), ('HOUSEHOLD_TYPE', 'int'), ('TEN', 'int')]

    geo_fields = ['TABBLKST', 'TABBLKCOU', 'TABTRACTCE', 'TABBLK', 'PLACEFP', 'AIANNHCE']
    geo_types = [row[1] for row in geo_df.dtypes if row[0] in geo_fields]
    assert geo_types == 6 * ['string']

    pop_types = [row for row in pop_df.dtypes]
    print('pop_types', pop_types)
    assert pop_types == [('REGION_ID','string'),('REGION_TYPE','string'),('ITERATION_CODE','string'),('COUNT','int')]

    print('output types and order test passed')

def test_pop_count():
    pop_cnt = 0
    reader = CEFHReader("src/phsafe_safetab_reader/safetab_h_cef_config_2010.ini", ['72', '01', '02'])
    pop_df_input = reader.get_pop_df_input()
    pop_df = reader.get_pop_counts()
    for cnt in pop_df:
       pop_cnt = pop_cnt + cnt
    assert pop_df_input.count() == pop_cnt

    print('test pop count passed')


def test_pop_aiannh_states():
    reader = CEFHReader("src/phsafe_safetab_reader/safetab_h_cef_config_2010.ini", ['72', '01', '02'])
    pop_df = reader.get_pop_group_details_df()
    geo_df = reader.get_geo_df()

    # filter pop_df to get only AIANNH entries
    pop_aiannh_df = pop_df.filter(col("REGION_TYPE").isin(["AIANNH"]))

    # join AIANNH entries ot GRFC to get associated state info
    aiannh_joined_to_grfc = pop_aiannh_df.join(geo_df, pop_aiannh_df.REGION_ID == geo_df.AIANNHCE, "leftouter")

    aiannh_states = [row.TABBLKST for row in aiannh_joined_to_grfc.select('TABBLKST').distinct().collect()]
    # Puerto Rico 72 should be absent because PR has no AIANNH areas
    assert set(aiannh_states) == set(['01', '02'])

    print('pop aiannh state test passed')

def test_pop_nonaiannh_states():
    reader = CEFHReader("src/phsafe_safetab_reader/safetab_h_cef_config_2010.ini", ['72', '01', '02'])
    pop_df = reader.get_pop_group_details_df()
    geo_df = reader.get_geo_df()

    # skip AIANNH regions and the national USA region
    pop_nonaiannh_df = pop_df.filter(~col("REGION_TYPE").isin(["AIANNH", "USA"]))

    non_aiannh_states = [row.STATESUBSTR for row in pop_nonaiannh_df.select(
        substring(pop_nonaiannh_df.REGION_ID, 1, 2).alias("STATESUBSTR")).distinct().collect()]
    assert set(non_aiannh_states) == set(['72', '01', '02'])

    print('pop non-aiannh state test passed')



def run_all_tests():
    test_pop_nonaiannh_states()
    test_pop_aiannh_states()
    test_pr_only()
    test_us_only()
    test_pr_and_us()
    test_allstates_and_pr()
    test_input_types()
    test_output_types_and_order()
    test_pop_count()

if __name__ == "__main__":

    run_all_tests()
