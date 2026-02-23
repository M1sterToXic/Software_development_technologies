import ctypes
import numpy as np
from ctypes import c_int, c_float, c_double, POINTER, byref

lib = ctypes.CDLL("/mnt/c/Users/krivo/Tecnology_PO/openblas/libopenblas.so.0")

CblasRowMajor = 101
CblasColMajor = 102
CblasNoTrans = 111
CblasTrans = 112
CblasConjTrans = 113
CblasUpper = 121
CblasLower = 122
CblasNonUnit = 131
CblasUnit = 132

lib.cblas_sgemv.argtypes = [c_int, c_int, c_int, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int, c_float, POINTER(c_float), c_int]
lib.cblas_dgemv.argtypes = [c_int, c_int, c_int, c_int, c_double, POINTER(c_double), c_int, POINTER(c_double), c_int, c_double, POINTER(c_double), c_int]
lib.cblas_strmv.argtypes = [c_int, c_int, c_int, c_int, c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]
lib.cblas_dtrmv.argtypes = [c_int, c_int, c_int, c_int, c_int, POINTER(c_double), c_int, POINTER(c_double), c_int]
lib.cblas_ssymv.argtypes = [c_int, c_int, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int, c_float, POINTER(c_float), c_int]
lib.cblas_dsymv.argtypes = [c_int, c_int, c_int, c_double, POINTER(c_double), c_int, POINTER(c_double), c_int, c_double, POINTER(c_double), c_int]
lib.cblas_sger.argtypes = [c_int, c_int, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]
lib.cblas_dger.argtypes = [c_int, c_int, c_int, c_double, POINTER(c_double), c_int, POINTER(c_double), c_int, POINTER(c_double), c_int]
lib.cblas_strsv.argtypes = [c_int, c_int, c_int, c_int, c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]
lib.cblas_dtrsv.argtypes = [c_int, c_int, c_int, c_int, c_int, POINTER(c_double), c_int, POINTER(c_double), c_int]
lib.cblas_stpmv.argtypes = [c_int, c_int, c_int, c_int, c_int, POINTER(c_float), POINTER(c_float), c_int]
lib.cblas_dtpmv.argtypes = [c_int, c_int, c_int, c_int, c_int, POINTER(c_double), POINTER(c_double), c_int]

def test_sgemv():
    print("Testing sgemv...")
    A = np.array([1, 2, 3, 4], dtype=np.float32)
    x = np.array([1, 1], dtype=np.float32)
    y = np.zeros(2, dtype=np.float32)
    expected = np.array([3, 7], dtype=np.float32)
    
    lib.cblas_sgemv(CblasRowMajor, CblasNoTrans, 2, 2, 1.0, A.ctypes.data_as(POINTER(c_float)), 2, x.ctypes.data_as(POINTER(c_float)), 1, 0.0, y.ctypes.data_as(POINTER(c_float)), 1)
    
    if np.allclose(y, expected, rtol=1e-6):
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: got {y}, expected {expected}")
        return False

def test_dgemv():
    print("Testing dgemv...")
    A = np.array([1, 2, 3, 4], dtype=np.float64)
    x = np.array([1, 1], dtype=np.float64)
    y = np.zeros(2, dtype=np.float64)
    expected = np.array([3, 7], dtype=np.float64)
    
    lib.cblas_dgemv(CblasRowMajor, CblasNoTrans, 2, 2, 1.0, A.ctypes.data_as(POINTER(c_double)), 2, x.ctypes.data_as(POINTER(c_double)), 1, 0.0, y.ctypes.data_as(POINTER(c_double)), 1)
    
    if np.allclose(y, expected, rtol=1e-12):
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: got {y}, expected {expected}")
        return False

def test_strmv():
    print("Testing strmv...")
    A = np.array([1, 2, 0, 4], dtype=np.float32)
    x = np.array([1, 1], dtype=np.float32)
    expected = np.array([3, 4], dtype=np.float32)
    
    lib.cblas_strmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, A.ctypes.data_as(POINTER(c_float)), 2, x.ctypes.data_as(POINTER(c_float)), 1)
    
    if np.allclose(x, expected, rtol=1e-6):
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: got {x}, expected {expected}")
        return False

def test_dtrmv():
    print("Testing dtrmv...")
    A = np.array([1, 2, 0, 4], dtype=np.float64)
    x = np.array([1, 1], dtype=np.float64)
    expected = np.array([3, 4], dtype=np.float64)
    
    lib.cblas_dtrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, A.ctypes.data_as(POINTER(c_double)), 2, x.ctypes.data_as(POINTER(c_double)), 1)
    
    if np.allclose(x, expected, rtol=1e-12):
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: got {x}, expected {expected}")
        return False

def test_ssymv():
    print("Testing ssymv...")
    A = np.array([1, 2, 2, 4], dtype=np.float32)
    x = np.array([1, 1], dtype=np.float32)
    y = np.zeros(2, dtype=np.float32)
    expected = np.array([3, 6], dtype=np.float32)
    
    lib.cblas_ssymv(CblasRowMajor, CblasUpper, 2, 1.0, A.ctypes.data_as(POINTER(c_float)), 2, x.ctypes.data_as(POINTER(c_float)), 1, 0.0, y.ctypes.data_as(POINTER(c_float)), 1)
    
    if np.allclose(y, expected, rtol=1e-6):
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: got {y}, expected {expected}")
        return False

def test_dsymv():
    print("Testing dsymv...")
    A = np.array([1, 2, 2, 4], dtype=np.float64)
    x = np.array([1, 1], dtype=np.float64)
    y = np.zeros(2, dtype=np.float64)
    expected = np.array([3, 6], dtype=np.float64)
    
    lib.cblas_dsymv(CblasRowMajor, CblasUpper, 2, 1.0, A.ctypes.data_as(POINTER(c_double)), 2, x.ctypes.data_as(POINTER(c_double)), 1, 0.0, y.ctypes.data_as(POINTER(c_double)), 1)
    
    if np.allclose(y, expected, rtol=1e-12):
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: got {y}, expected {expected}")
        return False

def test_sger():
    print("Testing sger...")
    A = np.zeros(4, dtype=np.float32)
    x = np.array([2, 3], dtype=np.float32)
    y = np.array([4, 5], dtype=np.float32)
    expected = np.array([8, 10, 12, 15], dtype=np.float32)
    
    lib.cblas_sger(CblasRowMajor, 2, 2, 1.0, x.ctypes.data_as(POINTER(c_float)), 1, y.ctypes.data_as(POINTER(c_float)), 1, A.ctypes.data_as(POINTER(c_float)), 2)
    
    if np.allclose(A, expected, rtol=1e-6):
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: got {A}, expected {expected}")
        return False

def test_dger():
    print("Testing dger...")
    A = np.zeros(4, dtype=np.float64)
    x = np.array([2, 3], dtype=np.float64)
    y = np.array([4, 5], dtype=np.float64)
    expected = np.array([8, 10, 12, 15], dtype=np.float64)
    
    lib.cblas_dger(CblasRowMajor, 2, 2, 1.0, x.ctypes.data_as(POINTER(c_double)), 1, y.ctypes.data_as(POINTER(c_double)), 1, A.ctypes.data_as(POINTER(c_double)), 2)
    
    if np.allclose(A, expected, rtol=1e-12):
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: got {A}, expected {expected}")
        return False

def test_strsv():
    print("Testing strsv...")
    A = np.array([1, 2, 0, 4], dtype=np.float32)
    x = np.array([3, 4], dtype=np.float32)
    expected = np.array([1, 1], dtype=np.float32)
    
    lib.cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, A.ctypes.data_as(POINTER(c_float)), 2, x.ctypes.data_as(POINTER(c_float)), 1)
    
    if np.allclose(x, expected, rtol=1e-6):
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: got {x}, expected {expected}")
        return False

def test_dtrsv():
    print("Testing dtrsv...")
    A = np.array([1, 2, 0, 4], dtype=np.float64)
    x = np.array([3, 4], dtype=np.float64)
    expected = np.array([1, 1], dtype=np.float64)
    
    lib.cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, A.ctypes.data_as(POINTER(c_double)), 2, x.ctypes.data_as(POINTER(c_double)), 1)
    
    if np.allclose(x, expected, rtol=1e-12):
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: got {x}, expected {expected}")
        return False

def test_stpmv():
    print("Testing stpmv...")
    Ap = np.array([1, 2, 4], dtype=np.float32)
    x = np.array([1, 1], dtype=np.float32)
    expected = np.array([3, 4], dtype=np.float32)
    
    lib.cblas_stpmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, Ap.ctypes.data_as(POINTER(c_float)), x.ctypes.data_as(POINTER(c_float)), 1)
    
    if np.allclose(x, expected, rtol=1e-6):
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: got {x}, expected {expected}")
        return False

def test_dtpmv():
    print("Testing dtpmv...")
    Ap = np.array([1, 2, 4], dtype=np.float64)
    x = np.array([1, 1], dtype=np.float64)
    expected = np.array([3, 4], dtype=np.float64)
    
    lib.cblas_dtpmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, Ap.ctypes.data_as(POINTER(c_double)), x.ctypes.data_as(POINTER(c_double)), 1)
    
    if np.allclose(x, expected, rtol=1e-12):
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: got {x}, expected {expected}")
        return False

def run_all_tests():
    print("\n=== CBLAS LEVEL 2 TESTS (Python) ===\n")
    
    tests = [
        test_sgemv,
        test_dgemv,
        test_strmv,
        test_dtrmv,
        test_ssymv,
        test_dsymv,
        test_sger,
        test_dger,
        test_strsv,
        test_dtrsv,
        test_stpmv,
        test_dtpmv
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== RESULT: {passed}/{len(tests)} tests passed ===")

if __name__ == "__main__":
    run_all_tests()
