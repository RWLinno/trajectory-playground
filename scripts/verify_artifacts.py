"""Verify provenance, reconstruct samples from weights, and replay held-out metrics."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.experiment import Regressor,Denoiser,sample,conditions,reference_amplitudes,paths,metrics

torch.set_num_threads(2)
report=json.loads((ROOT/'artifacts/evaluation.json').read_text())
bank=json.loads((ROOT/'docs/assets/samples.json').read_text())
for key,file in [('source_sha256','src/experiment.py'),('weights_sha256','artifacts/models.pt')]:
    assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==report['metadata'][key],file
checkpoint=torch.load(ROOT/'artifacts/models.pt',map_location='cpu',weights_only=True)
mlp,diffusion=Regressor(),Denoiser()
mlp.load_state_dict(checkpoint['mlp']);diffusion.load_state_dict(checkpoint['diffusion'])
mlp.eval();diffusion.eval()
rng=np.random.default_rng(20261014)
heldout=conditions(rng,128)
generator=torch.Generator().manual_seed(54321)
rows={'mlp':[],'diffusion':[],'reference':[]}
for i,c in enumerate([s['condition'] for s in bank['scenes']]+list(heldout)):
    cc=torch.tensor(np.tile(c,(64,1)),dtype=torch.float32)
    with torch.no_grad(): ma=mlp(cc)[:,0].numpy()
    generated={'mlp':ma,'diffusion':sample(diffusion,cc,generator),'reference':reference_amplitudes(rng,np.tile(c,(64,1)))}
    for name,amplitudes in generated.items():
        if i<9:
            np.testing.assert_allclose(amplitudes,bank['scenes'][i]['models'][name]['amplitudes'],atol=2e-5,rtol=0)
        else:
            actual=metrics(paths(amplitudes,c),c)
            expected=report['per_condition'][name][i-9]
            for key,value in actual.items():
                # CPU kernels differ between ARM/macOS and x86/Linux. Discrete
                # collision and coverage outcomes must still match exactly.
                tolerance = 1e-4 if key in ['mean_length','min_clearance'] else 0
                np.testing.assert_allclose(value,expected[key],atol=tolerance,rtol=0,err_msg=f'{name} condition {i-9} metric {key}')
            rows[name].append(actual)
for name,expected in report['summary'].items():
    for key,value in expected.items(): np.testing.assert_allclose(np.mean([r[key] for r in rows[name]]),value,atol=2e-5,rtol=0)
print('PASS: hashes, 9 complete sample banks, 128 held-out conditions, all three methods, summaries.')
