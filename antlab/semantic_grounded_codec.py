"""Experimental AVS1 shared-contract vector packets; default AVL1 unchanged."""
import math
import struct

HEADER=struct.Struct('<4sBBHI')


def anchors(width):
    if type(width) is not int or width not in (2,16):
        raise ValueError('unsupported anchor width')
    return [list(p)+[0.]*(width-2) for p in [(-.8,0.),(.8,0.),(0.,.8),(0.,-.8)]]


def _contract(contract,width):
    if type(contract) is not int or contract not in (0,1) or type(width) is not int or width not in (2,16) or (contract==0 and width!=16):
        raise ValueError('unsupported semantic contract/width')


def pack(vectors,contract):
    if type(vectors) is not list or not 1<=len(vectors)<=1024 or type(vectors[0]) is not list:
        raise ValueError('expected bounded vector list')
    width=len(vectors[0]); _contract(contract,width)
    if any(type(v) is not list or len(v)!=width or any(type(x) not in (int,float) or not math.isfinite(x) for x in v) for v in vectors):
        raise ValueError('expected rectangular finite numerical vectors')
    values=[x for v in vectors for x in v]
    try:
        return HEADER.pack(b'AVS1',1,contract,width,len(vectors))+struct.pack('<'+'f'*len(values),*values)
    except (OverflowError,struct.error) as exc:
        raise ValueError('values exceed float32 transport') from exc


def unpack(packet,expected_contract,expected_width):
    _contract(expected_contract,expected_width)
    if type(packet) is not bytes or len(packet)<HEADER.size:
        raise ValueError('truncated packet')
    magic,version,contract,width,count=HEADER.unpack_from(packet)
    _contract(contract,width)
    if (magic,version,contract,width)!=(b'AVS1',1,expected_contract,expected_width) or not 1<=count<=1024 or len(packet)!=HEADER.size+count*width*4:
        raise ValueError('packet framing or contract mismatch')
    values=struct.unpack_from('<'+'f'*(count*width),packet,HEADER.size)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('nonfinite received vector')
    return [list(values[i:i+width]) for i in range(0,len(values),width)]

