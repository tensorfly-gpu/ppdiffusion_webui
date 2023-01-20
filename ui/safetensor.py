import numpy as np

class safe_open():

    md_size = 0
    offset = 0
    metadata = b''
    _TYPES = {
        #"BF16": np.bfloat16,
        "F64": np.float64,
        "F32": np.float32,
        "F16": np.float16,
        "I64": np.int64,
        "U64": np.uint64,
        "I32": np.int32,
        "U32": np.uint32,
        "I16": np.int16,
        "U16": np.uint16,
        "I8": np.int8,
        "U8": np.uint8,
        "BOOL": np.bool,
    }

    def __init__(self, path):
        self.file = open(path, "rb") 

    def get_md_size(self):
        self.file.seek(0)
        self.md_size = int.from_bytes(self.file.read(8), 'little')
        self.offset = self.md_size+8

    def get_metadata(self):
        self.file.seek(8)
        self.metadata = self.file.read(self.md_size)
        self.metadata = eval(str(self.metadata, 'utf-8'))

    def keys(self):
        return self.metadata.keys()

    def get_type(self, tp):
        return self._TYPES[tp]

    def get_tensor(self, key):
        info = self.metadata[key]
        start = info["data_offsets"][0]
        size = info["data_offsets"][1]-start
        shape = info["shape"]
        d_type = self.get_type(info["dtype"])
        self.file.seek(self.offset+start)
        return np.frombuffer(self.file.read(size), dtype=d_type).reshape(shape)

