"""Re-run diagnostics for completed classes with fixed postprocess code."""
import sys, logging, pickle, numpy as np, torch, json, hashlib
sys.path.insert(0, 'D:/thesis_final/src')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')

from vae.schema import get_partition
from vae.model import MixedInputBetaVAE
from vae.dataset import PerClassDataset
from vae.diagnostics import run_diagnostics
from vae.config import DEFAULT_CONFIG

root = 'D:/thesis_final'
with open(root + '/data/processed/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
partition = get_partition()
X_val = np.load(root + '/data/processed/X_val.npy')
y_val_8 = np.load(root + '/data/processed/y_val_cat.npy')
config = DEFAULT_CONFIG


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


with open(root + '/vae_run_manifest.json') as f:
    manifest = json.load(f)

for class_id, class_name in [(0, 'Benign'), (1, 'BruteForce'), (2, 'DDoS'), (3, 'DoS')]:
    print(f'\n=== {class_name} ===')
    ckpt_path = root + f'/models/vae/vae_class_{class_id}_{class_name}.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = MixedInputBetaVAE(
        partition=partition,
        latent_dim=config['latent_dim'][class_name],
        n_pseudo_binary=0,
    )
    model.load_state_dict(ckpt['state_dict'])
    model.register_protocol_references(scaler)
    model.eval()
    val_ds = PerClassDataset(X_val, y_val_8, class_id=class_id, scaler=scaler, partition=partition)
    result = run_diagnostics(class_id, class_name, model, val_ds, scaler, partition, device='cpu')
    pc = result['posterior_collapse']['collapsed_dim_count']
    proto = result['per_feature_recon']['protocol_top1_accuracy']
    uncond = result['unconditional_validity']['overall_validity_rate']
    cond = result['conditional_validity']['overall_validity_rate']
    cons = result['unconditional_validity']['protocol_binary_consistency']
    print(f'  collapsed={pc}  proto_acc={proto:.4f}  uncond={uncond:.4f}  cond={cond:.4f}  consistency={cons}')
    uncond_block = result['unconditional_validity']
    cond_block = result['conditional_validity']
    manifest['diagnostics'][class_name] = {
        'path': root + f'/results/vae/diagnostics_{class_name}.json',
        'collapsed_dim_count': pc,
        'unconditional_validity_pre_postprocess': uncond_block.get('pre_postprocess_validity_rate'),
        'unconditional_validity_raw': uncond_block.get('overall_validity_rate_raw'),
        'unconditional_validity': uncond_block.get('overall_validity_rate'),
        'unconditional_validity_postprocess': uncond_block.get('overall_validity_rate_postprocess'),
        'unconditional_postprocess_repair_rate': uncond_block.get('postprocess_repair_rate'),
        'conditional_validity_pre_postprocess': cond_block.get('pre_postprocess_validity_rate'),
        'conditional_validity_raw': cond_block.get('overall_validity_rate_raw'),
        'conditional_validity': cond_block.get('overall_validity_rate'),
        'conditional_validity_postprocess': cond_block.get('overall_validity_rate_postprocess'),
        'conditional_postprocess_repair_rate': cond_block.get('postprocess_repair_rate'),
        'protocol_accuracy': proto,
    }
    manifest['checkpoints'][class_name] = {
        'path': ckpt_path,
        'sha256': sha256(ckpt_path),
        'best_val_loss': float(ckpt.get('best_val_loss', 0.0)),
        'epochs_trained': int(ckpt.get('epoch', 0)),
        'n_train': int(manifest.get('checkpoints', {}).get(class_name, {}).get('n_train', 0)),
        'n_val': len(val_ds),
        'latent_dim': config['latent_dim'].get(class_name, 16) if isinstance(config['latent_dim'], dict) else int(config['latent_dim']),
    }

with open(root + '/vae_run_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print('\nManifest updated.')
