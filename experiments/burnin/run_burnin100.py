"""Retrain 20 seeds with 100 warm-up items, retaining the matched zero baseline."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
from run_burnin_control import ROOT,OUT,train_one


def main():
    config=json.loads((OUT/'config.json').read_text())
    assert config['rho']==.95 and config['sigma']==.05
    assert config['epochs']==4000 and config['n_test']==2000
    assert config['seeds']==list(range(42,62))
    for seed in config['seeds']:
        meta=json.loads((OUT/'zero'/f'seed{seed}'/'complete.json').read_text())
        assert meta['burnin']==0 and meta['epochs']==4000 and meta['test_per_position']==2000
    config.update({'train_burnin':100,'evaluation_burnins':[100],
                   'zero_baseline':'previous matched zero-start runs, reused without retraining',
                   'workers':4,'variance_deficit_percent_after_100':100*.95**200,
                   'source_archive_runner':'run_burnin_control.py',
                   'runner_sha256':hashlib.sha256((ROOT/'run_burnin_control.py').read_bytes()).hexdigest()})
    (OUT/'config_burnin100.json').write_text(json.dumps(config,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(train_one,s,100) for s in config['seeds']]
        for f in as_completed(fs):f.result()
    for seed in config['seeds']:
        a=json.loads((OUT/'zero'/f'seed{seed}'/'complete.json').read_text())
        b=json.loads((OUT/'burnin100'/f'seed{seed}'/'complete.json').read_text())
        assert a['training_input_checksum']==b['training_input_checksum']
    print('ALL 20 BURN-IN-100 MODELS COMPLETE; measured training inputs match zero baseline.',flush=True)


if __name__=='__main__':main()
