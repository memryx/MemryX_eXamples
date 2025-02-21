"""
mxpack.py
The MxPack serialization format and utilities.



Copyright (c) 2024 MemryX Inc.
MIT License

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from io import BytesIO
import struct
import numpy as np
from os import PathLike


##########################################################################################
# CORE API
##########################################################################################

def MxPackEncode(v):
    """
    Encodes a Python List or Dict into the MxPack serialization format.

    Returns a bytearray() with the encoded data.


    Parameters
    ----------
    v: list or dict
        The data to be encoded

    """
    if type(v) is dict:
        r = bytearray()
        r.extend(b'\x01')
        r = _write_dict_no_header(v, r)
        return r
    elif type(v) is list:
        r = bytearray()
        r.extend(b'\x02')
        r = _write_list_no_header(v, r)
        return r
    else:
        raise TypeError(f"MxPackEncode cannot encode type {type(v)}, only list and dict")
        return None


def MxPackDecode(v):
    """
    Decodes a MxPack bytes/bytearray/BytesIO object into a Python Dict or List.

    Also accepts a string or pathlike to a file to decode (testing only).

    Returns a a list[] or dict{} with the decoded data.


    Parameters
    ----------
    v: bytes, bytearray, BytesIO, str, or os.PathLike object
        The data to be decoded

    """
    ret = None
    if type(v) in [str, PathLike]:
        with open(v, 'rb') as f:
            dat = BytesIO(f.read())
            t = dat.read(1)
            if t == b'\x01':
                ret = _parse_dict(dat)
            elif t == b'\x02':
                ret = _parse_list(dat, False)
            else:
                raise ValueError(f"MxPackDecode got unknown outer dtype byte: {t}")
    elif type(v) in [bytes, bytearray]:
        dat = BytesIO(v)
        t = dat.read(1)
        if t == b'\x01':
            ret = _parse_dict(dat)
        elif t == b'\x02':
            ret = _parse_list(dat, False)
        else:
            raise ValueError(f"MxPackDecode got unknown outer dtype byte: {t}")
    elif type(v) is BytesIO:
        t = v.read(1)
        if t == b'\x01':
            ret = _parse_dict(v)
        elif t == b'\x02':
            ret = _parse_list(v, False)
        else:
            raise ValueError(f"MxPackDecode got unknown outer dtype byte: {t}")
    else:
        raise TypeError(f"MxPackDecode got unsupport input object type: {type(v)}")

    return ret


##########################################################################################
# DECODING HELPERS
##########################################################################################

def _parse_dict(b : BytesIO):

    ret = {}
    num_keys = int.from_bytes(b.read(4), byteorder='little')

    for _ in range(num_keys):
        keyname = str(b.read(64).decode('ascii')).split("\x00")[0]
        dtype = int.from_bytes(b.read(1), byteorder='little')

        if dtype == 0x70:
            theb = int.from_bytes(b.read(1), byteorder='little', signed=False)
            if theb == 0:
                ret[keyname] = False
            else:
                ret[keyname] = True
        elif dtype == 0x10:
            ret[keyname] = int.from_bytes(b.read(1), byteorder='little', signed=False)
        elif dtype == 0x11:
            ret[keyname] = int.from_bytes(b.read(1), byteorder='little', signed=True)
        elif dtype == 0x20:
            ret[keyname] = int.from_bytes(b.read(2), byteorder='little', signed=False)
        elif dtype == 0x21:
            ret[keyname] = int.from_bytes(b.read(2), byteorder='little', signed=True)
        elif dtype == 0x30:
            ret[keyname] = int.from_bytes(b.read(4), byteorder='little', signed=False)
        elif dtype == 0x31:
            ret[keyname] = int.from_bytes(b.read(4), byteorder='little', signed=True)
        elif dtype == 0x40:
            ret[keyname] = int.from_bytes(b.read(8), byteorder='little', signed=False)
        elif dtype == 0x41:
            ret[keyname] = int.from_bytes(b.read(8), byteorder='little', signed=True)
        elif dtype == 0x50:
            ret[keyname] = float(struct.unpack('<f', bytearray(b.read(4)))[0])
        elif (dtype == 0x03) or (dtype == 0x61):
            binsize = int.from_bytes(b.read(8), byteorder='little', signed=False)
            ret[keyname] = bytearray(b.read(binsize))
        elif dtype == 0x60:
            strleng = int.from_bytes(b.read(4), byteorder='little', signed=False)
            ret[keyname] = str(b.read(strleng).decode('ascii')).split("\x00")[0]
        elif dtype == 0x01:
            ret[keyname] = _parse_dict(b)
        elif dtype == 0x02:
            ret[keyname] = _parse_list(b, False)
        elif dtype == 0xA2:
            ret[keyname] = _parse_list(b, True)
        else:
            raise ValueError(f"MxPack::_parse_dict got invalid dtype byte: {dtype}")
            return None

    return ret


def _parse_list(b : BytesIO,
                is_numpy : bool):

    ret = []

    dtype = int.from_bytes(b.read(1), byteorder='little')
    num_elem = int.from_bytes(b.read(4), byteorder='little', signed=False)

    if dtype == 0x70:
        for _ in range(num_elem):
            theb = int.from_bytes(b.read(1), byteorder='little', signed=False)
            if theb == 0:
                ret.append(False)
            else:
                ret.append(True)
    elif dtype == 0x10:
        for _ in range(num_elem):
            ret.append(int.from_bytes(b.read(1), byteorder='little', signed=False))
        if is_numpy:
            ret = np.asarray(ret, order='C', dtype=np.uint8)
    elif dtype == 0x11:
        for _ in range(num_elem):
            ret.append(int.from_bytes(b.read(1), byteorder='little', signed=True))
        if is_numpy:
            ret = np.asarray(ret, order='C', dtype=np.int8)
    elif dtype == 0x20:
        for _ in range(num_elem):
            ret.append(int.from_bytes(b.read(2), byteorder='little', signed=False))
        if is_numpy:
            ret = np.asarray(ret, order='C', dtype=np.uint16)
    elif dtype == 0x21:
        for _ in range(num_elem):
            ret.append(int.from_bytes(b.read(2), byteorder='little', signed=True))
        if is_numpy:
            ret = np.asarray(ret, order='C', dtype=np.int16)
    elif dtype == 0x30:
        for _ in range(num_elem):
            ret.append(int.from_bytes(b.read(4), byteorder='little', signed=False))
        if is_numpy:
            ret = np.asarray(ret, order='C', dtype=np.uint32)
    elif dtype == 0x31:
        for _ in range(num_elem):
            ret.append(int.from_bytes(b.read(4), byteorder='little', signed=True))
        if is_numpy:
            ret = np.asarray(ret, order='C', dtype=np.int32)
    elif dtype == 0x40:
        for _ in range(num_elem):
            ret.append(int.from_bytes(b.read(8), byteorder='little', signed=False))
        if is_numpy:
            ret = np.asarray(ret, order='C', dtype=np.uint64)
    elif dtype == 0x41:
        for _ in range(num_elem):
            ret.append(int.from_bytes(b.read(8), byteorder='little', signed=True))
        if is_numpy:
            ret = np.asarray(ret, order='C', dtype=np.int64)
    elif dtype == 0x50:
        for _ in range(num_elem):
            ret.append(float(struct.unpack('<f', bytearray(b.read(4)))[0]))
        if is_numpy:
            ret = np.asarray(ret, order='C', dtype=np.float32)
    elif dtype == 0x60:
        for _ in range(num_elem):
            strleng = int.from_bytes(b.read(4), byteorder='little', signed=False)
            ret.append(str(b.read(strleng).decode('ascii')).split("\x00")[0])
    elif (dtype == 0x03) or (dtype == 0x61):
        for _ in range(num_elem):
            binsize = int.from_bytes(b.read(8), byteorder='little', signed=False)
            ret.append(bytearray(b.read(binsize)))
    elif dtype == 0x02:
        for _ in range(num_elem):
            ret.append(_parse_list(b, False))
    elif dtype == 0xA2:
        for _ in range(num_elem):
            ret.append(_parse_list(b, True))
        ret = np.asarray(ret, order='C')
    elif dtype == 0x01:
        for _ in range(num_elem):
            ret.append(_parse_dict(b))
    else:
        raise ValueError(f"MxPack::_parse_list got invalid dtype byte: {dtype}")
        return None

    return ret


##########################################################################################
# ENCODING HELPERS
##########################################################################################

def _write_dict_no_header(wd : dict,
                         r  : bytearray):
    numkeys = int(len(wd.items()))
    r.extend(numkeys.to_bytes(4, byteorder='little'))
    for key,val in wd.items():
        keystr = str(key)
        keystbytes = bytes(keystr, 'ascii')
        keystbytelen = int(len(keystbytes))
        if keystbytelen < 64:
            r.extend(keystbytes)
            k = keystbytelen
            # pad to 64 chars
            while k < 64:
                r.extend(b'\x00')
                k += 1
        else:
            raise ValueError(f"MxPack::_write_dict: dict key string is too long ({keystbytelen} > 63)")
            return None

        # key done, now do the value
        r = _write_val_with_header(val, r)

    return r


def _write_list_no_header(wl : list,
                         r  : bytearray):
    # get length
    leng = len(wl)

    # element dtype
    mtype = 0x00
    if isinstance(wl, np.ndarray):
        if type(wl[0]) in [list, np.ndarray]:
            mtype = 0xA2
        elif wl.dtype == np.uint8:
            mtype = 0x10
            wl = list(map(int, wl.tolist()))
        elif wl.dtype == np.int8:
            mtype = 0x11
            wl = list(map(int, wl.tolist()))
        elif wl.dtype == np.uint16:
            mtype = 0x20
            wl = list(map(int, wl.tolist()))
        elif wl.dtype == np.int16:
            mtype = 0x21
            wl = list(map(int, wl.tolist()))
        elif wl.dtype == np.uint32:
            mtype = 0x30
            wl = list(map(int, wl.tolist()))
        elif wl.dtype == np.int32:
            mtype = 0x31
            wl = list(map(int, wl.tolist()))
        elif wl.dtype == np.uint64:
            mtype = 0x40
            wl = list(map(int, wl.tolist()))
        elif wl.dtype == np.int64:
            mtype = 0x41
            wl = list(map(int, wl.tolist()))
        elif wl.dtype in [float, np.float16, np.float32, np.float64]:
            mtype = 0x50
            wl = list(map(float, wl.tolist()))
        else:
            raise ValueError(f"MxPack::_write_list: got a numpy ndarray with unknown dtype {wl.dtype}")
            return None
    elif type(wl[0]) is int:
        neg = False
        biggest = int(0)
        for n in wl:
            if abs(n) > biggest:
                biggest = abs(n)
            if n < 0:
                neg = True

        if biggest == 0:
            numb = 1
        else:
            numb = int(np.ceil(biggest.bit_length() / 8.0))
        if (numb == 1) and neg:
            mtype = 0x11
        elif (numb == 1) and not neg:
            mtype = 0x10
        elif (numb == 2) and neg:
            mtype = 0x21
        elif (numb == 2) and not neg:
            mtype = 0x20
        elif (numb == 3 or numb == 4) and neg:
            mtype = 0x31
        elif (numb == 3 or numb == 4) and not neg:
            mtype = 0x30
        elif (numb > 4 and numb <= 8) and neg:
            mtype = 0x41
        elif (numb > 4 and numb <= 8) and not neg:
            mtype = 0x40
        else:
            raise ValueError(f"MxPack::_write_list: integer of magnitude {biggest} is not representable")
            return None
    elif type(wl[0]) in [float, np.float16, np.float32, np.float64]:
        mtype = 0x50
    elif type(wl[0]) is str:
        mtype = 0x60
    elif type(wl[0]) is bool:
        mtype = 0x70
    elif type(wl[0]) is dict:
        mtype = 0x01
    elif type(wl[0]) is list:
        mtype = 0x02
    elif type(wl[0]) is np.ndarray:
        mtype = 0xA2
    elif type(wl[0]) in [bytes, bytearray]:
        mtype = 0x03
    else:
        raise TypeError(f"MxPack::_write_list got data of invalid type {type(wl[0])}")
        return None

    # write the dtype
    r.extend(mtype.to_bytes(1, byteorder='little'))

    # write the array length
    r.extend(leng.to_bytes(4, byteorder='little'))

    # write the elements
    for n in wl:
        if mtype == 0x70:
            if n == False:
                r.extend(b'\x00')
            else:
                r.extend(b'\x01')
        elif (mtype & 0xF0) == 0x10:
            v = int(n)
            r.extend(v.to_bytes(1, byteorder='little', signed=(v<0)))
        elif (mtype & 0xF0) == 0x20:
            v = int(n)
            r.extend(v.to_bytes(2, byteorder='little', signed=(v<0)))
        elif (mtype & 0xF0) == 0x30:
            v = int(n)
            r.extend(v.to_bytes(4, byteorder='little', signed=(v<0)))
        elif (mtype & 0xF0) == 0x40:
            v = int(n)
            r.extend(v.to_bytes(8, byteorder='little', signed=(v<0)))
        elif mtype == 0x50:
            r.extend(bytearray(struct.pack('<f',float(n))))
        elif mtype == 0x60:
            stbytes = bytearray(n, 'ascii')
            stbytes.extend(b'\x00')
            stlen = int(len(stbytes))
            r.extend(stlen.to_bytes(4, byteorder='little'))
            r.extend(stbytes)
        elif mtype == 0x03:
            blen = int(len(n))
            r.extend(blen.to_bytes(8, byteorder='little'))
            r.extend(n)
        elif mtype == 0x02:
            r = _write_list_no_header(n, r)
        elif mtype == 0xA2:
            r = _write_list_no_header(n, r)
        elif mtype == 0x01:
            r = _write_dict_no_header(n, r)
        else:
            raise RuntimeError(f"MxPack::_write_list somehow got an mtype of {mtype}")
            return None

    return r



def _write_val_with_header(i,
                          r : bytearray):
    if type(i) in [int, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
        i = int(i)
        if(i == 0):
            numb = 1
        else:
            numb = int(np.ceil(i.bit_length() / 8.0))
        neg = (i < 0)
        if (numb == 1) and neg:
            r.extend(b'\x11')
            r.extend(i.to_bytes(1, byteorder='little', signed=True))
        elif (numb == 1) and not neg:
            r.extend(b'\x10')
            r.extend(i.to_bytes(1, byteorder='little'))
        elif (numb == 2) and neg:
            r.extend(b'\x21')
            r.extend(i.to_bytes(2, byteorder='little', signed=True))
        elif (numb == 2) and not neg:
            r.extend(b'\x20')
            r.extend(i.to_bytes(2, byteorder='little'))
        elif (numb == 3 or numb == 4) and neg:
            r.extend(b'\x31')
            r.extend(i.to_bytes(4, byteorder='little', signed=True))
        elif (numb == 3 or numb == 4) and not neg:
            r.extend(b'\x30')
            r.extend(i.to_bytes(4, byteorder='little'))
        elif (numb > 4 and numb <= 8) and neg:
            r.extend(b'\x41')
            r.extend(i.to_bytes(8, byteorder='little', signed=True))
        elif (numb > 4 and numb <= 8) and not neg:
            r.extend(b'\x40')
            r.extend(i.to_bytes(8, byteorder='little'))
        else:
            raise ValueError(f"MxPack::_write_val_with_header: integer of magnitude {i} is not representable")
            return None
    elif type(i) is bool:
        r.extend(b'\x70')
        if i == False:
            r.extend(b'\x00')
        else:
            r.extend(b'\x01')
    elif type(i) in [float, np.float32, np.float16, np.float64]:
        r.extend(b'\x50')
        r.extend(bytearray(struct.pack('<f',float(i))))
    elif type(i) is str:
        r.extend(b'\x60')
        stbytes = bytearray(i, 'ascii')
        stbytes.extend(b'\x00')
        stlen = int(len(stbytes))
        r.extend(stlen.to_bytes(4, byteorder='little'))
        r.extend(stbytes)
    elif type(i) in [bytes, bytearray]:
        r.extend(b'\x03')
        blen = int(len(i))
        r.extend(blen.to_bytes(8, byteorder='little'))
        r.extend(i)
    elif type(i) is dict:
        r.extend(b'\x01')
        r = _write_dict_no_header(i, r)
    elif type(i) is list:
        r.extend(b'\x02')
        r = _write_list_no_header(i, r)
    elif type(i) is np.ndarray:
        r.extend(b'\xA2')
        r = _write_list_no_header(i, r)
    else:
        raise TypeError(f"MxPack::_write_val_with_header got data of invalid type {type(i)}")
        return None


    return r
