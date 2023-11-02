import ctools.geo as geo
import pytest


@pytest.mark.parametrize(
    "name, dir_name, state_name, abbr, fips",[
        ('alabama', 'Alabama/', 'Alabama', 'AL', '01'),
        ('Alaska', 'Alaska/', 'Alaska', 'AK', '02'),
        ('AZ', 'Arizona/', 'Arizona', 'AZ', '04'),
    ])
def test_state_rec_name(name, dir_name, state_name, abbr, fips):
    data = geo.state_rec(name)
    assert data['dir_name'] == dir_name
    assert data['state_name'] == state_name
    assert data['state_abbr'] == abbr
    assert data['fips_state'] == fips


def test_state_rec_fips():
    fips_code = '01'
    data = geo.state_rec(fips=fips_code)
    assert data['state_abbr'] == 'AL'
    assert data['fips_state'] == '01'


def test_state_rec_fail():
    fake_state = 'ZZ'
    fake_fip = '90'
    with pytest.raises(ValueError):
        data = geo.state_rec(fake_state)
    with pytest.raises(ValueError):
        data = geo.state_rec(fake_fip)


def test_conversions():
    assert geo.state_fips('AL') == '01'
    assert geo.state_fips('Alaska') == '02'
    assert geo.state_abbr('04') == 'AZ'
    # this list includes DC ... so 51
    assert len(geo.all_state_abbrs()) == 51
