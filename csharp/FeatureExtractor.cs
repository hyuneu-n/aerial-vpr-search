using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.Versioning;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VprSearch
{
    // 단일 이미지 -> DINOv3 CLS 토큰(1024차원, L2 정규화) 추론기.
    // 배치 전처리/추론 파이프라인(인덱스 빌드용)은 이 저장소 범위 밖이라 제외.
    [SupportedOSPlatform("windows")]
    public class FeatureExtractor : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly int _dim = 1024;
        private readonly int _imgSize;

        // SAT-493M 항공 이미지 전용 정규화 파라미터
        private readonly float[] _mean = { 0.430f, 0.411f, 0.296f };
        private readonly float[] _std = { 0.213f, 0.156f, 0.143f };

        public FeatureExtractor(string modelPath, int imgSize = 512)
        {
            _imgSize = imgSize;
            var options = new SessionOptions();

            try
            {
                options.AppendExecutionProvider_CUDA(0);
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            }
            catch
            {
                Console.WriteLine("[WARN] CUDA 가속 실패 → CPU 모드");
            }

            _session = new InferenceSession(modelPath, options);
            _inputName = _session.InputMetadata.Keys.First();
        }

        // 단일 이미지 추론 (Search 단계에서 사용)
        public float[] ExtractSingle(string jpgPath)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, _imgSize, _imgSize });
            using (Bitmap bmp = new Bitmap(jpgPath))
            {
                FillTensorFromBitmap(bmp, tensor, 0);
            }

            var inputs = new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
            using var results = _session.Run(inputs);
            var output = results.First().AsTensor<float>().ToArray();
            NormalizeInPlace(output);
            return output;
        }

        // 이미지 전처리: 리사이즈 + SAT-493M 정규화 + 텐서 변환
        private void FillTensorFromBitmap(Bitmap source, DenseTensor<float> tensor, int batchIdx)
        {
            using (Bitmap resized = new Bitmap(_imgSize, _imgSize))
            using (Graphics g = Graphics.FromImage(resized))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.DrawImage(source, 0, 0, _imgSize, _imgSize);

                BitmapData data = resized.LockBits(
                    new Rectangle(0, 0, _imgSize, _imgSize),
                    ImageLockMode.ReadOnly,
                    PixelFormat.Format24bppRgb);

                int imageOffset = batchIdx * 3 * _imgSize * _imgSize;
                int channelSize = _imgSize * _imgSize;

                unsafe
                {
                    byte* ptr = (byte*)data.Scan0;
                    Span<float> dest = tensor.Buffer.Span;

                    for (int y = 0; y < _imgSize; y++)
                    {
                        for (int x = 0; x < _imgSize; x++)
                        {
                            int i = y * data.Stride + x * 3;

                            // BGR → RGB 변환 + SAT-493M 정규화
                            dest[imageOffset + 0 * channelSize + y * _imgSize + x] =
                                ((ptr[i + 2] / 255f) - _mean[0]) / _std[0]; // R
                            dest[imageOffset + 1 * channelSize + y * _imgSize + x] =
                                ((ptr[i + 1] / 255f) - _mean[1]) / _std[1]; // G
                            dest[imageOffset + 2 * channelSize + y * _imgSize + x] =
                                ((ptr[i + 0] / 255f) - _mean[2]) / _std[2]; // B
                        }
                    }
                }

                resized.UnlockBits(data);
            }
        }

        // L2 정규화: 벡터 크기를 1로 만들어 코사인 유사도 계산 가능하게 함
        private void NormalizeInPlace(float[] v)
        {
            double sumSq = 0;
            for (int i = 0; i < v.Length; i++)
                sumSq += (double)v[i] * v[i];

            float norm = (float)Math.Sqrt(sumSq);
            if (norm < 1e-6f)
                return;

            for (int i = 0; i < v.Length; i++)
                v[i] /= norm;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
