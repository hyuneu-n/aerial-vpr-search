import numpy as np
import os
import sys

# Environment detection (Docker / Jetson / Windows DLL / Others)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JETSON_LIB_PATH = "/app/libs/libfaiss_c.so"
WINDOWS_DLL_PATH = os.path.join(PROJECT_ROOT, "libs", "faiss_c.dll")

IS_JETSON = os.path.exists(JETSON_LIB_PATH) or os.path.exists('/sys/devices/gpu.0/load')
IS_WINDOWS_DLL = (os.name == 'nt') and os.path.exists(WINDOWS_DLL_PATH)

# Jetson / Windows(동봉된 DLL 있음) 에서는 C-API 기반 엔진,
# 그 외(WLS/리눅스/윈도우 일반)에서는 Python FAISS 바인딩을 사용합니다.

if IS_JETSON or IS_WINDOWS_DLL:
    # -------------------------------------------------------------------------
    # Jetson: C-API 기반 FAISS (ctypes)
    # -------------------------------------------------------------------------
    import ctypes
    
    class FaissNativePy:
        def __init__(self, lib_path=None):
            # 라이브러리 경로 자동 선택
            if lib_path is None:
                if IS_JETSON:
                    lib_path = JETSON_LIB_PATH
                elif IS_WINDOWS_DLL:
                    lib_path = WINDOWS_DLL_PATH
                else:
                    lib_path = JETSON_LIB_PATH  # 최후의 수단

            if not os.path.exists(lib_path):
                raise FileNotFoundError(f"❌ 라이브러리 부재: {lib_path}")
            
            self.lib = ctypes.CDLL(lib_path)

            # C-API 함수 규격 정의
            self.lib.faiss_StandardGpuResources_new.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            self.lib.faiss_StandardGpuResources_new.restype = ctypes.c_int

            self.lib.faiss_read_index_fname.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
            self.lib.faiss_read_index_fname.restype = ctypes.c_int

            self.lib.faiss_index_cpu_to_gpu.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            self.lib.faiss_index_cpu_to_gpu.restype = ctypes.c_int

            self.lib.faiss_Index_search.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p]
            self.lib.faiss_Index_search.restype = ctypes.c_int

            self.lib.faiss_Index_ntotal.argtypes = [ctypes.c_void_p]
            self.lib.faiss_Index_ntotal.restype = ctypes.c_long
            
            self.lib.faiss_Index_d.argtypes = [ctypes.c_void_p]
            self.lib.faiss_Index_d.restype = ctypes.c_int

            self.lib.faiss_Index_free.argtypes = [ctypes.c_void_p]
            self.lib.faiss_GpuResources_free.argtypes = [ctypes.c_void_p]

    class FaissEngine:
        """Jetson용 FAISS Engine (C-API 기반, GPU)"""
        def __init__(self, path=None, dim=1024):
            self.d = dim
            self.native = FaissNativePy()
            self.res_ptr = ctypes.c_void_p()
            self.native.lib.faiss_StandardGpuResources_new(ctypes.byref(self.res_ptr))
            self.index_ptr = None
            self.ref_labels = []
            self.index = type('IndexObj', (), {'d': self.d})()

            if path:
                self.load(path)

        def load(self, path):
            # A. 인덱스 파일 로드 (CPU)
            cpu_idx_ptr = ctypes.c_void_p()
            path_bytes = (path + "\0").encode('utf-8')
            ret = self.native.lib.faiss_read_index_fname(path_bytes, 0, ctypes.byref(cpu_idx_ptr))
            if ret != 0:
                raise RuntimeError(f"Index load failed: {path}")

            # B. 차원 동기화
            self.d = self.native.lib.faiss_Index_d(cpu_idx_ptr)
            self.index.d = self.d

            # C. GPU로 전송
            self.index_ptr = ctypes.c_void_p()
            self.native.lib.faiss_index_cpu_to_gpu(self.res_ptr, 0, cpu_idx_ptr, ctypes.byref(self.index_ptr))
            self.native.lib.faiss_Index_free(cpu_idx_ptr)

            # D. 라벨 파일 로드
            label_path = path.replace(".index", ".txt")
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    self.ref_labels = [line.strip() for line in f.readlines()]
                print(f"✅ GPU Index Ready (C-API): {path} (Labels: {len(self.ref_labels)})")
            else:
                print(f"⚠️ 라벨 파일을 찾을 수 없습니다: {label_path}. ID를 라벨로 사용합니다.")
                self.ref_labels = []

        def search(self, query_vector, k=5):
            if self.index_ptr is None:
                raise RuntimeError("인덱스가 로드되지 않았습니다!")
                
            query_vector = np.ascontiguousarray(query_vector.astype('float32'))
            distances = np.zeros((1, k), dtype='float32')
            labels = np.zeros((1, k), dtype='int64')
            
            self.native.lib.faiss_Index_search(
                self.index_ptr, 1, 
                query_vector.ctypes.data, k, 
                distances.ctypes.data, labels.ctypes.data
            )
            return distances, labels

else:
    # -------------------------------------------------------------------------
    # WSL / 리눅스 / 윈도우: Python FAISS (GPU 있으면 GPU, 없으면 CPU)
    # -------------------------------------------------------------------------
    try:
        import faiss
        HAS_FAISS = True
        HAS_FAISS_GPU = hasattr(faiss, "StandardGpuResources")
    except ImportError:
        HAS_FAISS = False
        HAS_FAISS_GPU = False
        print("⚠️ FAISS 파이썬 패키지가 설치되어 있지 않습니다. (예: pip install faiss-gpu 또는 faiss-cpu)")

    class FaissEngine:
        """WSL/리눅스/윈도우용 FAISS Engine (GPU/CPU 자동 선택)"""
        def __init__(self, path=None, dim=1024):
            if not HAS_FAISS:
                raise ImportError("FAISS 파이썬 패키지가 필요합니다. (faiss-gpu 또는 faiss-cpu)")
            
            self.d = dim
            self.gpu_index = None
            self.cpu_index = None
            self.ref_labels = []
            self.index = type('IndexObj', (), {'d': self.d})()
            
            # GPU 리소스 (가능한 경우에만)
            self.res = faiss.StandardGpuResources() if HAS_FAISS_GPU else None

            if path:
                self.load(path)

        def load(self, path):
            # A. CPU 인덱스 로드
            cpu_index = faiss.read_index(path)
            self.d = cpu_index.d
            self.index.d = self.d

            # B. GPU 사용 가능하면 GPU로 전송, 아니면 CPU 그대로 사용
            if HAS_FAISS_GPU and self.res is not None:
                self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
                self.cpu_index = None
                engine_type = "Python FAISS GPU"
            else:
                self.cpu_index = cpu_index
                self.gpu_index = None
                engine_type = "Python FAISS CPU"

            # C. 라벨 파일 로드
            label_path = path.replace(".index", ".txt")
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    self.ref_labels = [line.strip() for line in f.readlines()]
                print(f"✅ Index Ready ({engine_type}): {path} (Labels: {len(self.ref_labels)})")
            else:
                print(f"⚠️ 라벨 파일을 찾을 수 없습니다: {label_path}")

        def search(self, query_vector, k=5):
            if self.gpu_index is None and self.cpu_index is None:
                raise RuntimeError("인덱스가 로드되지 않았습니다!")
            
            query_vector = np.ascontiguousarray(query_vector.astype('float32'))
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            if self.gpu_index is not None:
                distances, labels = self.gpu_index.search(query_vector, k)
            else:
                distances, labels = self.cpu_index.search(query_vector, k)

            return distances, labels