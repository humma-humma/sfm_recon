"""Make the local research checkout's attention mask representable in FP16."""
import argparse
import hashlib
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--gnt',type=Path,required=True)
parser.add_argument('--restore',action='store_true')
args=parser.parse_args()
path=args.gnt/'gnt/transformer_network.py'
source=path.read_text(encoding='utf-8')
old='attn = attn.masked_fill(mask == 0, -1e9)'
new='attn = attn.masked_fill(mask == 0, max(-1e9, torch.finfo(attn.dtype).min))'
if args.restore:
    assert old in source or new in source
    path.write_text(source.replace(new,old),encoding='utf-8',newline='\n')
elif old in source:
    assert source.count(old)==1
    path.write_text(source.replace(old,new),encoding='utf-8')
else:
    assert new in source, 'Unexpected GNT attention implementation; preserving it'
print('GNT dtype-safe attention mask:',hashlib.sha256(path.read_bytes()).hexdigest())
