"""Descriptive seed-level mechanism evidence, after the complete formal audit.

This is an exploratory supplement, not a change to frozen success statistics.
Contacts are geometric, nominal P is not measured physical pitch, and endpoints
of failed movements can differ from endpoints of completed tasks.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from pitch_adaptation_experiment_v5 import ROOT, SCENES
from pitch_adaptation_formal_v5 import BASE, verify_freeze
from pitch_adaptation_statistics import distribution


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_trial(row, analysis):
    case = ROOT / row['source_directory']
    folder = analysis / 'mechanisms' / case.name
    summary = read(folder / 'mechanism.json')
    for name, expected in summary['sources_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Mechanism input changed: ' + name)
    if (summary['scene'], summary['method'], summary['seed']) != (row['scene'], row['method'], row['seed']):
        raise ValueError('Mechanism case identity differs')
    result = read(case / 'result.json')
    with np.load(folder / 'mechanism.npz', allow_pickle=False) as data:
        marker = result.get('hold_start_s')
        if marker is None:
            marker = 2.2 + row['motion_s']
        index = int(np.argmin(abs(data['sim_time'] - marker)))
        gap = float(abs(data['sim_time'][index] - marker))
        if gap > .0100001:
            raise ValueError('Mechanism lacks a physical movement-end sample')
        lengths = data['body_length_m'][index]
        turns = data['signed_turns'][index]
        contacts = data['geometric_contacts'][index]
        modules = data['contact_modules'][index]
        field = int(np.searchsorted(data['field_time'], marker + 1e-8, side='right') - 1)
        if field < 0:
            raise ValueError('No applied control field before movement end')
        alpha = np.degrees(data['alpha_rad'][field])
        pitch = data['pitch_mm'][field]
        finite_pitch = pitch[np.isfinite(pitch)]
        output = dict(scene=row['scene'], method=row['method'], seed=row['seed'], outcome=row['outcome'],
                      success=row['success'], hold_started=result.get('hold_start_s') is not None,
                      marker_sim_time_s=float(marker), marker_sample_gap_s=gap,
                      end_total_net_turns=float(abs(turns.sum())),
                      end_terminal_region_net_turns=float(abs(turns[-1])),
                      end_geometric_contacts=int(contacts.sum()),
                      end_contact_modules=int(np.any(modules, axis=0).sum()),
                      end_actual_module_radius_mean_mm=float(data['actual_module_radius_mm'][index].mean()),
                      end_control_alpha_spread_deg=float(np.ptp(alpha)),
                      end_control_alpha_median_deg=float(np.median(alpha)),
                      end_nominal_pitch_median_mm=float(np.median(finite_pitch)) if len(finite_pitch) else None,
                      end_nominal_pitch_finite_knots=int(len(finite_pitch)),
                      accepted_window_radius_rmse_mm=summary['accepted_radius_window_rmse_mm'],
                      accepted_pointwise_radius_rmse_mm=summary['accepted_radius_pointwise_rmse_mm'],
                      maximum_pointwise_radius_error_mm=summary['accepted_radius_pointwise_abs_error_max_mm'],
                      maximum_local_alpha_spread_deg=row['maximum_local_alpha_spread_deg'],
                      radius_valid_fraction=row['radius_valid_fraction'], fallback_node_fraction=row['fallback_node_fraction'])
        output['straddling_sample_frames'] = summary['straddling_frames']
        for region, values in summary['straddling'].items():
            output[region + '_straddle_body_length_m'] = values['median_body_length_m']
            output[region + '_straddle_net_turns'] = values['median_net_wrap_turns']
            output[region + '_straddle_geometric_contacts'] = values['median_geometric_contacts']
        for i, region in enumerate(data['regions']):
            output[region + '_end_length_m'] = float(lengths[i])
            output[region + '_end_signed_turns'] = float(turns[i])
            output[region + '_end_geometric_contacts'] = int(contacts[i])
        if abs(float(lengths.sum()) - 1.3545) > 1e-10:
            raise ValueError('Endpoint body length is not conserved')
    output['mechanism_json_sha256'] = digest(folder / 'mechanism.json')
    output['mechanism_npz_sha256'] = digest(folder / 'mechanism.npz')
    return output


METRICS = ('end_total_net_turns', 'end_terminal_region_net_turns', 'end_geometric_contacts', 'end_contact_modules',
           'end_actual_module_radius_mean_mm', 'end_control_alpha_spread_deg', 'end_control_alpha_median_deg',
           'end_nominal_pitch_median_mm', 'accepted_window_radius_rmse_mm', 'accepted_pointwise_radius_rmse_mm',
           'maximum_pointwise_radius_error_mm', 'maximum_local_alpha_spread_deg', 'radius_valid_fraction', 'fallback_node_fraction',
           'upstream_straddle_body_length_m', 'downstream_straddle_body_length_m',
           'upstream_straddle_net_turns', 'downstream_straddle_net_turns',
           'upstream_straddle_geometric_contacts', 'downstream_straddle_geometric_contacts')


def plot(groups, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [('end_total_net_turns', 'Net body turns at movement end'),
              ('end_geometric_contacts', 'Geometric contacts at movement end'),
              ('end_control_alpha_spread_deg', 'Control alpha spatial spread (deg)'),
              ('accepted_window_radius_rmse_mm', 'Accepted-window radius RMSE (mm)'),
              ('accepted_pointwise_radius_rmse_mm', 'Accepted-point radius RMSE (mm)'),
              ('fallback_node_fraction', 'Fallback node fraction during movement')]
    lookup = {(g['scene'], g['method']): g for g in groups}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, panels):
        for j, method in enumerate('ABCD'):
            values = [lookup[s, method]['metrics'][metric] for s in SCENES]
            middle = np.array([np.nan if v['median'] is None else v['median'] for v in values])
            lower = np.array([np.nan if v['q25'] is None else v['q25'] for v in values])
            upper = np.array([np.nan if v['q75'] is None else v['q75'] for v in values])
            ax.errorbar(np.arange(4) + (j - 1.5) * .17, middle, yerr=[middle - lower, upper - middle], fmt='o', capsize=3, label=method)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(np.arange(4), ['Small', 'Large', 'Horizontal taper', 'Vertical taper'], rotation=15)
        ax.grid(axis='y', alpha=.25)
    axes.flat[0].legend(ncol=4)
    fig.suptitle('Exploratory mechanism descriptions: 10 seed-level trials per group, median / IQR\nAll outcomes retained; movement endpoints may differ in task progress. Contacts do not establish support.', fontsize=12)
    fig.savefig(out / 'mechanism_comparison.png', dpi=150)
    plt.close(fig)


def main(analysis):
    verify_freeze(BASE / 'training_audit/parameter_freeze.json')
    audit = read(analysis / 'formal_audit.json')
    expected = {(s, m, seed) for s in SCENES for m in 'ABCD' for seed in range(101, 111)}
    actual = [(r['scene'], r['method'], r['seed']) for r in audit['rows']]
    if not audit.get('complete') or not audit.get('statistics_complete') or audit.get('mechanism_cases') != 160 or len(actual) != 160 or set(actual) != expected:
        raise ValueError('Descriptive mechanism comparison requires the complete audited 160-case batch and mechanisms')
    rows = [extract_trial(row, analysis) for row in audit['rows']]
    with (analysis / 'mechanism_by_seed.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        names = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader(); writer.writerows(rows)
    groups = []
    for scene in SCENES:
        for method in 'ABCD':
            subset = [r for r in rows if r['scene'] == scene and r['method'] == method]
            groups.append(dict(scene=scene, method=method, trials=len(subset), hold_started=sum(r['hold_started'] for r in subset),
                               metrics={metric: distribution(subset, metric) for metric in METRICS}))
    report = dict(exploratory_descriptive_only=True, unit='one scalar per paired seed trial; no frame-level inference',
                  endpoint='hold start if present, otherwise movement stop; all outcomes retained and progress may differ',
                  straddling='per-trial time medians while body occupies both pipe regions; varied transit durations, descriptive only',
                  radius='accepted fitted radius versus actual module/window radius about the truth axis; truth used only offline',
                  contacts='geometric positions, not force or stable support',
                  nominal_pitch='equivalent P from recorded radius/alpha; fixed-angle or fallback samples may be finite; not actual measured pitch',
                  radius_rmse_weighting='accepted module-fit calls within each trial, including repeated physical timestamps; ten seed-level scalars per group',
                  timing='supplement developed after inspecting partial formal records; exploratory only, primary statistics unchanged',
                  primary_statistics_unchanged=True, formal_audit_sha256=digest(analysis / 'formal_audit.json'),
                  source_sha256=digest(Path(__file__)), methods=groups, rows=rows)
    (analysis / 'mechanism_comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    plot(groups, analysis)
    print('Saved descriptive seed-level mechanism comparison for all 160 trials.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=BASE / 'formal_analysis')
    args = parser.parse_args(); main(args.analysis)
