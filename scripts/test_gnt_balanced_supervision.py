"""Test 90/10 original/added supervision against the completed matched arms."""
import argparse
from pathlib import Path
import torch
from test_gnt_dense_supervision import prepare,fit,evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    experiment,source,plan,teacher=prepare(args.root)
    checkpoints={arm:experiment/f'models/{arm}/splatfacto/continued/nerfstudio_models/step-000011999.ckpt' for arm in ('control','dense')}
    assert all(p.is_file() for p in checkpoints.values())
    checkpoints['balanced90']=fit(args.root,experiment,source,plan,'balanced90')
    torch.cuda.empty_cache()
    evaluate(args.root,source,checkpoints,plan,teacher)


if __name__=='__main__':main()
