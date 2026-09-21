"""Write a deterministic provenance inventory for the publication revision."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rel(path): return path.relative_to(ROOT).as_posix()


def main():
    src = sorted((ROOT / 'results/tc').glob('*_tc.json'))
    cohort, excluded = [], []
    for f in src:
        d = json.loads(f.read_text())
        row = {k:d[k] for k in ('seed','entries','bands','n_test','S_training_max',
                               'S_training_entries_above_one','S_heldout_exceedances','prediction_dtypes')}
        row.update(path=rel(f), sha256=sha(f),
                   nonpositive_transmission_entries=round(d['transmission']['fraction_nonpositive']*d['entries']),
                   families=sorted(d['families']))
        if d['seed'] in (104,105,106,107):
            if d['float32_opt_in'] or any(v != ['float64'] for v in d['prediction_dtypes'].values()):
                raise ValueError('Mixed precision in the common cohort')
            cohort.append(row)
        else:
            row['exclusion_reason']='Different, explicitly precision-altered float32 experiment'
            excluded.append(row)
    if [r['seed'] for r in cohort] != [104,105,106,107]:
        raise ValueError('The transmission table requires the four declared records')
    if len({tuple(r['families']) for r in cohort}) != 1 or len(cohort[0]['families']) != 9:
        raise ValueError('Heads differ across the common transmission cohort')
    patterns={
        'multi_kernel':'results/e2bc/mk_emit_*.json',
        'regularization':'results/e2bc/rs_emit_*.json',
        'feature_perturbation':'results/ridge/*_featpert.json',
        'ridge_path':'results/ridge/*_ridge.json',
        'oco2_selection':'results/oco2/*_select.json',
        'matched_benchmark_lanes':'results/pkanrtm/pkanrtm_*.json',
        'matched_benchmark_rescores':'results/pkanrtm/rescored*.json'}
    groups={k:[{'path':rel(f),'sha256':sha(f)} for f in sorted(ROOT.glob(p))] for k,p in patterns.items()}
    kkt=[json.loads((ROOT/r['path']).read_text())['rules']['stack_radiance']['kkt_gap']
         for r in groups['oco2_selection']]
    syn_path=ROOT/'results/sharpness_synthetic.json'
    syn=json.loads(syn_path.read_text())
    out={
        'schema_version':1,
        'source_manuscript_commit':'9184f0e6f910bd51e290747807a525749f28f588',
        'revision_scope':'Manuscript, source-contract corrections, public-record reanalysis and synthetic rerun; no trained-model rerun',
        'common_float64_transmission_cohort':cohort,
        'excluded_precision_experiments':excluded,
        'legacy_diagnostic_limit':'Violation mask is s<=S only; all-band averages include nonpositive t and possible invalid albedo. No full-domain or arbitrary-precision theorem certificate is inferred.',
        'record_groups':groups,
        'record_counts':{k:len(v) for k,v in groups.items()},
        'oco2_quadratic_kkt_gap_range':[min(kkt),max(kkt)],
        'synthetic':{'path':rel(syn_path),'sha256':sha(syn_path),
                     'sampled_transmissions':sum(v['n'] for v in syn['betas']),
                     'violations':sum(v['pointwise_bound_violations'] for v in syn['betas']),
                     'betas':[v['beta'] for v in syn['betas']]},
        'retained_historical_aggregates':[
            {'table':'tab:v2-replication','source':'paper/table_v2_replication.tex','missing':'results/e2a/emit_s*_big3.json'},
            {'table':'tab:v2-stacks','source':'paper/table_v2_stacks.tex','missing':'results/stack/*_stack.json'},
            {'table':'tab:v2-perturbation, first fourteen rows','source':'paper/table_v2_perturbation.tex','missing':'results/kernel_perturbation_Y2_n2000.json'}],
        'training_reproduction_gaps':['raw EMIT arrays','full-precision sample-level prediction archives',
            'simulator version and generation configuration','redistribution permissions',
            'complete individual records for some secondary ablations'],
        'numeric_tables':{p.name:sha(p) for p in sorted((ROOT/'paper').glob('table_v2_*.tex'))}}
    dest=ROOT/'results/revision_20260921/reanalysis.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('Wrote',rel(dest));print('record counts:',out['record_counts'])
    print('synthetic transmissions / violations:',out['synthetic']['sampled_transmissions'],out['synthetic']['violations'])

if __name__=='__main__': main()
