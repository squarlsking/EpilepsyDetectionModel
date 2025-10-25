import os
import struct

# 错误码 =
MTAR_ESUCCESS    = 0
MTAR_EFAILURE    = -1
MTAR_EOPENFAIL   = -2
MTAR_EREADFAIL   = -3
MTAR_EWRITEFAIL  = -4
MTAR_ESEEKFAIL   = -5
MTAR_EBADCHKSUM  = -6
MTAR_ENULLRECORD = -7
MTAR_ENOTFOUND   = -8

MTAR_TREG = '0'  # 普通文件
MTAR_TDIR = '5'  # 目录

#  header 结构
class MtarHeader:
    def __init__(self):
        self.name = ""
        self.mode = 0
        self.owner = 0
        self.size = 0
        self.mtime = 0
        self.type = MTAR_TREG
        self.linkname = ""


#  Tar 操作类
class MicroTar:
    def __init__(self, filename, mode="r"):
        self.filename = filename
        self.mode = mode
        self.pos = 0
        self.remaining_data = 0
        self.last_header = 0

        # 确保模式是二进制
        if "r" in mode:
            mode = "rb"
        elif "w" in mode:
            mode = "wb"
        elif "a" in mode:
            mode = "ab"

        try:
            self.fp = open(filename, mode)
        except:
            raise IOError("Could not open file")

    def close(self):
        self.fp.close()
        return MTAR_ESUCCESS

    def seek(self, pos):
        try:
            self.fp.seek(pos)
            self.pos = pos
            return MTAR_ESUCCESS
        except:
            return MTAR_ESEEKFAIL

    def read(self, size):
        data = self.fp.read(size)
        if len(data) != size:
            return None, MTAR_EREADFAIL
        self.pos += size
        return data, MTAR_ESUCCESS

    def write(self, data: bytes):
        try:
            self.fp.write(data)
            self.pos += len(data)
            return MTAR_ESUCCESS
        except:
            return MTAR_EWRITEFAIL

    def round_up(self, n, incr=512):
        return n + (incr - n % incr) % incr

    # Header 处理
    def _checksum(self, raw):
        # raw 是 512 字节 tar header
        chksum = 256  # 因为 checksum 字段先当作全空格
        for i in range(0, 148):
            chksum += raw[i]
        for i in range(156, 512):
            chksum += raw[i]
        return chksum

    def _raw_to_header(self, raw: bytes):
        if raw[148] == 0:
            return None, MTAR_ENULLRECORD

        h = MtarHeader()
        h.name = raw[0:100].decode("utf-8").rstrip("\0")
        h.mode = int(raw[100:108].decode("utf-8").strip() or "0", 8)
        h.owner = int(raw[108:116].decode("utf-8").strip() or "0", 8)
        h.size = int(raw[124:136].decode("utf-8").strip() or "0", 8)
        h.mtime = int(raw[136:148].decode("utf-8").strip() or "0", 8)
        h.type = chr(raw[156]) if raw[156] != 0 else MTAR_TREG
        h.linkname = raw[157:257].decode("utf-8").rstrip("\0")

        # 校验和检查
        stored_sum = int(raw[148:156].decode("utf-8").strip() or "0", 8)
        calc_sum = self._checksum(raw)
        if stored_sum != calc_sum:
            return None, MTAR_EBADCHKSUM

        return h, MTAR_ESUCCESS

    def _header_to_raw(self, h: MtarHeader):
        raw = bytearray(512)
        raw[0:100] = h.name.encode("utf-8").ljust(100, b"\0")
        raw[100:108] = ("%07o" % h.mode).encode("utf-8")
        raw[108:116] = ("%07o" % h.owner).encode("utf-8")
        raw[124:136] = ("%011o" % h.size).encode("utf-8")
        raw[136:148] = ("%011o" % h.mtime).encode("utf-8")
        raw[156:157] = h.type.encode("utf-8")
        raw[157:257] = h.linkname.encode("utf-8").ljust(100, b"\0")

        # 写 checksum
        raw[148:156] = b"        "
        chksum = self._checksum(raw)
        raw[148:156] = ("%06o\0 " % chksum).encode("utf-8")
        return bytes(raw)

    # 读写接口
    def read_header(self):
        self.last_header = self.pos
        raw, err = self.read(512)
        if err != MTAR_ESUCCESS:
            return None, err
        self.seek(self.last_header)  # 回到 header 开始
        return self._raw_to_header(raw)

    def next(self):
        h, err = self.read_header()
        if err != MTAR_ESUCCESS:
            return err
        size = self.round_up(h.size, 512) + 512
        return self.seek(self.pos + size)

    def find(self, name):
        self.seek(0)
        while True:
            h, err = self.read_header()
            if err != MTAR_ESUCCESS:
                if err == MTAR_ENULLRECORD:
                    return None, MTAR_ENOTFOUND
                return None, err
            if h.name == name:
                return h, MTAR_ESUCCESS
            self.next()

    def read_data(self, size):
        if self.remaining_data == 0:
            h, err = self.read_header()
            if err != MTAR_ESUCCESS:
                return None, err
            self.seek(self.pos + 512)
            self.remaining_data = h.size
        data, err = self.read(size)
        if err != MTAR_ESUCCESS:
            return None, err
        self.remaining_data -= size
        if self.remaining_data == 0:
            self.seek(self.last_header)
        return data, MTAR_ESUCCESS

    def write_header(self, h: MtarHeader):
        raw = self._header_to_raw(h)
        self.remaining_data = h.size
        return self.write(raw)

    def write_file_header(self, name, size):
        h = MtarHeader()
        h.name = name
        h.size = size
        h.type = MTAR_TREG
        h.mode = 0o664
        return self.write_header(h)

    def write_dir_header(self, name):
        h = MtarHeader()
        h.name = name
        h.type = MTAR_TDIR
        h.mode = 0o775
        return self.write_header(h)

    def write_data(self, data: bytes):
        err = self.write(data)
        if err != MTAR_ESUCCESS:
            return err
        self.remaining_data -= len(data)
        if self.remaining_data == 0:
            padding = self.round_up(self.pos, 512) - self.pos
            self.write(b"\0" * padding)
        return MTAR_ESUCCESS

    def finalize(self):
        self.write(b"\0" * 1024)
        return MTAR_ESUCCESS
