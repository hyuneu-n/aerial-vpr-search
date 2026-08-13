using System;
using System.Runtime.InteropServices;

namespace VprSearch
{
    // FAISS C API의 검색측 서브셋만 노출: 인덱스 로드/검색/파라미터 설정.
    // 인덱스 생성(factory는 환경 점검용으로만 사용)/학습/저장(add, train, write)은 포함하지 않음.
    public static class FaissNative
    {
        private const string DllName = "faiss_c.dll";

        // 시스템에서 인식된 NVIDIA GPU 개수 반환 (0이면 CPU 모드)
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int faiss_get_num_gpus(ref int p_out);

        // 인덱스 팩토리: check 모드에서 FAISS 라이브러리 로드 확인용으로만 사용
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int faiss_index_factory(out IntPtr p_index, int d, string description, int metric);

        // 유사도 검색: 쿼리 벡터에 대해 가장 가까운 상위 K개의 거리와 인덱스 반환
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int faiss_Index_search(IntPtr index, long n, float[] x, long k, float[] distances, long[] labels);

        // 생성된 인덱스 메모리 해제 (필수)
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void faiss_Index_free(IntPtr index);

        // 저장된 인덱스 파일을 CPU 메모리로 로드 (반드시 _fname 버전 사용)
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int faiss_read_index_fname(byte[] fname, int io_flags, out IntPtr p_out);

        // ParameterSpace: IVF nprobe 등 검색 파라미터 설정
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int faiss_ParameterSpace_new(out IntPtr p_space);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void faiss_ParameterSpace_free(IntPtr space);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int faiss_ParameterSpace_set_index_parameter(
            IntPtr space,
            IntPtr index,
            [MarshalAs(UnmanagedType.LPStr)] string name,
            double value);

        public static int GetGpuCount()
        {
            int count = 0;
            faiss_get_num_gpus(ref count);
            return count;
        }

        // 간단한 헬퍼 함수 (ParameterSpace 없이 직접 설정, 예: nprobe=10)
        public static void SetIndexParameter(IntPtr index, string name, double value)
        {
            faiss_ParameterSpace_new(out IntPtr space);
            try
            {
                faiss_ParameterSpace_set_index_parameter(space, index, name, value);
            }
            finally
            {
                faiss_ParameterSpace_free(space);
            }
        }
    }
}
