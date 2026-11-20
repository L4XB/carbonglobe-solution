"""Spin-up-only diagnostic of input variants on one test site (copy of work/edv3/diag_variants.py with the
deflate / norepair / tamp084 variants): VARIANT=<tag> python scripts/ed/diag_spin_variants.py <site>."""
import sys; sys.path.insert(0, 'scripts/ed'); sys.path.insert(0, 'src')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import make_site_inputs as msi, run_site as rs
SITE = int(sys.argv[1]) if len(sys.argv) > 1 else 1344
def spin(item):
    tag, over = item
    root = Path('work/edv3/diag_%s_%d' % (tag, SITE)); root.mkdir(parents=True, exist_ok=True)
    args = rs.config_args('b'); args.out = str(root / 'inputs')
    for k, v in over.items(): setattr(args, k, v)
    floors = msi.rsds_channel_floors()
    m = msi.build_site(SITE, 'ssp126', args, floors); msi.build_site(SITE, 'ssp585', args, floors, latlon=(m['lat_est'], m['lon_est']))
    inp = {sc: root / 'inputs' / f'site{SITE:05d}_{sc}' for sc in ('ssp126', 'ssp585')}
    run = root / 'run'; run.mkdir(exist_ok=True); (run / 'out').mkdir(exist_ok=True)
    rs.write_cfgs(run, inp, rs.EDV3 / 'EDv3_code' / 'ED_params.defaults.cfg', rs.EDV3 / 'EDv3_code' / 'ED_pft.defaults.cfg', fire_on=bool(over.get('fire_on', False)))
    rs.run_ed(run, 'spinup', 'models_spin.cfg', {'ED_DUMP_YEAR_SUFFIX': '1'}, run / 'spinup.log')
    tabs = {k: rs.parse_ed(v) for k, v in rs.site_files(run / 'out' / 'spinup', 'spinup').items()}
    rep = dict(m['repairs']); rep['spinup'] = str(m['spinup_climatology'])
    return tag, m['y0_decoded'], {a: rs.state(tabs, float(a)) for a in (10, 30, 70, 100, 200, 300, 500)}, rep
if __name__ == '__main__':
    import os
    variant = os.environ.get('VARIANT', 'c400f')
    base = {'repair': True, 'co2': 'real', 'co2_spinup': 'fixed', 'co2_spinup_fixed': 400.0, 'fire_on': True}
    variants = {'c400f': {}, 'geotwin': {'fire_on': False, 'spinup_geo_twin': True}, 'deflate': {'fire_on': False, 'deflate': True}, 'norepair': {'fire_on': False, 'repair': False}, 'tamp084': {'fire_on': False, 'ta_amp_factor': 0.84}, 'shiftp15': {'fire_on': False, 'spinup_shift': 15.0}, 'shiftp45': {'fire_on': False, 'spinup_shift': 45.0}, 'shiftp60': {'fire_on': False, 'spinup_shift': 60.0}, 'pr05': {'pr_scale': 0.5}, 'rh1': {'hus_cap_rh': 1.0}, 'ksat10': {'ksat_scale': 10.0},
                'co2stored': {'co2': 'asis', 'co2_spinup': 'first'}, 'pr07': {'pr_scale': 0.7}, 'twin': {'spinup_twin': True},
                'nofire': {'fire_on': False}, 'shiftm10': {'fire_on': False, 'spinup_shift': -10.0}, 'shiftp30': {'fire_on': False, 'spinup_shift': 30.0}}
    jobs = {variant: {**base, **variants[variant]}}
    with ProcessPoolExecutor(1) as ex:
        res = list(ex.map(spin, list(jobs.items())))
    ages = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    for tag, y0, out, rep in res:
        print('=== site %d spin-up, %s (%s) ===' % (SITE, tag, '; '.join(k for k in rep)))
        for a, s in out.items():
            i = ages.index(a)
            print('   age %3d  height %5.2f / y0 %5.2f | agb %5.2f / y0 %5.2f | lai(Jan) %4.2f / y0 %4.2f' % (a, s['height_m'], y0['height'][i], s['agb_kgC'], y0['agb'][i], s['lai'], y0['lai'][i]))
