using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Runtime.Versioning;

namespace VprSearch
{
    // JSON 구조체
    public class ImageMetadata
    {
        public string imageref { get; set; } = "";
        public Position position { get; set; } = new();
        public Attitude attitude { get; set; } = new();
    }

    public class Position
    {
        public double latitude { get; set; }
        public double longitude { get; set; }
        public double altitude { get; set; }
    }

    public class Attitude
    {
        public double yaw { get; set; }
    }

    public class MetadataFile
    {
        public List<ImageMetadata> images { get; set; } = new();
    }

    // ============================================================
    // Search-only entry point.
    // Index building (gallery -> DINO features -> FAISS .index/.txt)
    // is intentionally NOT part of this repo. Bring your own
    // model.onnx, faiss_c.dll and pre-built .index/.txt files.
    // ============================================================
    [SupportedOSPlatform("windows")]
    internal class Program
    {
        static void Main(string[] args)
        {
            string root = Directory.GetCurrentDirectory();
            string modelPath = Path.Combine(root, "model", "model.onnx");
            string queryDir = Path.Combine(root, "TestData", "query");

            string refJson = Path.Combine(root, "reference_metadata.json");
            string queryJson = Path.Combine(root, "query_metadata.json");

            int imgSize = 256;
            string indexArg = "Flat";

            if (args.Length == 0) { PrintUsage(); return; }
            string mode = args[0].ToLower();

            for (int i = 1; i < args.Length; i++)
            {
                if (args[i] == "--query" && i + 1 < args.Length) queryDir = args[++i];
                else if (args[i] == "--imgsize" && i + 1 < args.Length) imgSize = int.Parse(args[++i]);
                else if (args[i] == "--index" && i + 1 < args.Length) indexArg = args[++i];
                else if (args[i] == "--ref-json" && i + 1 < args.Length) refJson = args[++i];
                else if (args[i] == "--query-json" && i + 1 < args.Length) queryJson = args[++i];
            }

            // "Flat" 같은 타입 이름이 오면 dino_temp\<타입>.index 를 찾고,
            // ".index"가 붙었거나 경로 구분자가 있으면 그대로 파일 경로로 사용.
            string tempDir = Path.Combine(root, "dino_temp");
            string indexPath = (indexArg.EndsWith(".index", StringComparison.OrdinalIgnoreCase) ||
                                 indexArg.Contains(Path.DirectorySeparatorChar))
                ? indexArg
                : Path.Combine(tempDir, $"{indexArg.Replace(",", "_")}.index");
            string mapPath = Path.ChangeExtension(indexPath, ".txt");

            try
            {
                if (mode == "check") RunCheck(modelPath);
                else if (mode == "search") SearchAndValidate(refJson, queryJson, modelPath, indexPath, mapPath, queryDir, imgSize);
                else Console.WriteLine($"Unknown mode: {mode}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n[ERROR] {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }

        static void PrintUsage()
        {
            Console.WriteLine("Usage: dotnet run -- <check|search> [options]");
            Console.WriteLine("\nOptions:");
            Console.WriteLine("  --query <path>        쿼리 이미지 폴더");
            Console.WriteLine("  --imgsize <N>         이미지 해상도 (default: 256)");
            Console.WriteLine("  --index <type/path>   Flat 등 타입명 또는 .index 파일 경로");
            Console.WriteLine("  --ref-json <path>     Reference JSON");
            Console.WriteLine("  --query-json <path>   Query JSON");
            Console.WriteLine("\n(인덱스 빌드는 이 저장소 범위 밖입니다. 미리 빌드된 .index/.txt를 준비하세요.)");
        }

        static void RunCheck(string modelPath)
        {
            Console.WriteLine("\n=== FAISS-GPU 환경 점검 ===\n");
            try
            {
                Console.Write("[1/3] FAISS 라이브러리... ");
                FaissNative.faiss_index_factory(out IntPtr idx, 128, "Flat", 0);
                FaissNative.faiss_Index_free(idx);
                Console.WriteLine("OK");

                Console.Write("[2/3] GPU 감지... ");
                int gpuCount = FaissNative.GetGpuCount();
                Console.WriteLine(gpuCount > 0 ? $"OK ({gpuCount}개)" : "⚠ CPU 모드");

                Console.Write("[3/3] ONNX Runtime... ");
                using (var _ = new Microsoft.ML.OnnxRuntime.InferenceSession(modelPath))
                    Console.WriteLine("OK");

                Console.WriteLine("\n✅ 시스템 정상\n");
            }
            catch (Exception ex) { Console.WriteLine($"\n❌ 실패: {ex.Message}"); }
        }

        // ============================================================
        // SearchAndValidate: 단일 인덱스 파일 로드 → 쿼리별 Top-5 검색 →
        // 위경도(Haversine) 기준 Geo Top-5와 비교해 Hit@1/Hit@5 산출
        // ============================================================
        static void SearchAndValidate(string refJsonPath, string queryJsonPath, string model,
            string idxPath, string mapPath, string queryDir, int size)
        {
            Console.WriteLine("\n=== [Single Index Mode] Visual vs Geo Validation ===\n");
            var sw = Stopwatch.StartNew();

            var refMeta = LoadMetadata(refJsonPath);
            var queryMeta = LoadMetadata(queryJsonPath);

            Console.WriteLine($"Reference Meta: {refMeta.Count}개");
            Console.WriteLine($"Query Meta: {queryMeta.Count}개");

            if (!File.Exists(idxPath) || !File.Exists(mapPath))
            {
                Console.WriteLine($"\n[ERROR] 파일을 찾을 수 없습니다!");
                Console.WriteLine($"Index: {idxPath}");
                Console.WriteLine($"Map  : {mapPath}");
                return;
            }

            Console.WriteLine($"\n[Load] Index: {Path.GetFileName(idxPath)}");
            Console.WriteLine($"[Load] Map  : {Path.GetFileName(mapPath)}");

            FaissNative.faiss_read_index_fname(Encoding.UTF8.GetBytes(idxPath + '\0'), 0, out IntPtr idx);

            var refFileNames = File.ReadAllLines(mapPath).ToList();
            Console.WriteLine($"인덱스 데이터 크기: {refFileNames.Count}장");

            var orderedRefMeta = new List<ImageMetadata>();
            foreach (var path in refFileNames)
            {
                string fname = Path.GetFileName(path);
                var meta = refMeta.FirstOrDefault(m => Path.GetFileName(m.imageref) == fname);
                orderedRefMeta.Add(meta ?? new ImageMetadata { imageref = fname });
            }

            // IVF 인덱스인 경우 nprobe 설정 (Flat이면 무시됨)
            try { FaissNative.SetIndexParameter(idx, "nprobe", 10); } catch { }

            var queries = Directory.EnumerateFiles(queryDir, "*.*", SearchOption.AllDirectories)
                .Where(s => s.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                            s.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                .ToList();

            var results = new List<string> { "Query,Visual_Rank,Match_File,Similarity,Distance_m,Geo_Rank,In_Geo_Top5" };
            int top1Hit = 0, top5Hit = 0, totalQueries = 0;

            var validRefMeta = orderedRefMeta.Where(m => m.position.latitude != 0).ToList();

            using (var extractor = new FeatureExtractor(model, size))
            {
                int processed = 0;
                foreach (var qPath in queries)
                {
                    string qFileName = Path.GetFileName(qPath);
                    var qData = queryMeta.FirstOrDefault(m => Path.GetFileName(m.imageref) == qFileName);
                    if (qData == null) continue;

                    float[] feat = extractor.ExtractSingle(qPath);

                    const int K = 5;
                    float[] dists = new float[K];
                    long[] labels = new long[K];
                    FaissNative.faiss_Index_search(idx, 1, feat, K, dists, labels);

                    var candidates = new List<(string FileName, float Score, ImageMetadata Meta)>();
                    for (int k = 0; k < K; k++)
                    {
                        long id = labels[k];
                        if (id < 0 || id >= refFileNames.Count) continue;
                        candidates.Add((refFileNames[(int)id], dists[k], orderedRefMeta[(int)id]));
                    }
                    if (candidates.Count == 0) continue;

                    var geoTop5 = validRefMeta
                        .Select(m => new { Data = m, Dist = Haversine(qData, m) })
                        .OrderBy(x => x.Dist).Take(5)
                        .Select(x => Path.GetFileName(x.Data.imageref)).ToHashSet();

                    if (geoTop5.Contains(Path.GetFileName(candidates[0].FileName))) top1Hit++;

                    bool hasMatch = candidates.Any(c => geoTop5.Contains(Path.GetFileName(c.FileName)));
                    if (hasMatch) top5Hit++;

                    for (int k = 0; k < candidates.Count; k++)
                    {
                        var cand = candidates[k];
                        string matchName = Path.GetFileName(cand.FileName);
                        double actDist = Haversine(qData, cand.Meta);
                        bool inGeo = geoTop5.Contains(matchName);
                        int geoRank = -1; // 생략 가능

                        results.Add($"{qFileName},{k + 1},{matchName},{cand.Score:F6},{actDist:F1},{geoRank},{inGeo}");
                    }

                    totalQueries++;
                    processed++;
                    Console.Write($"\rProgress: {processed}/{queries.Count} (Acc: {((double)top1Hit / processed * 100):F1}%)   ");
                }
            }

            FaissNative.faiss_Index_free(idx);

            string outDir = Path.Combine(Directory.GetCurrentDirectory(), "result");
            Directory.CreateDirectory(outDir);
            string outPath = Path.Combine(outDir, $"single_idx_val_{DateTime.Now:MMdd_HHmmss}.csv");
            File.WriteAllLines(outPath, results);

            Console.WriteLine($"\n\n=== Validation Results ===");
            Console.WriteLine($"Queries: {totalQueries}");
            Console.WriteLine($"Hit@1 Rate: {(double)top1Hit / totalQueries * 100:F1}%");
            Console.WriteLine($"Hit@5 Rate: {(double)top5Hit / totalQueries * 100:F1}%");
            Console.WriteLine($"Saved: {outPath}\n");
        }

        static List<ImageMetadata> LoadMetadata(string jsonPath)
        {
            var json = File.ReadAllText(jsonPath);
            var data = JsonSerializer.Deserialize<MetadataFile>(json);
            return data?.images ?? new List<ImageMetadata>();
        }

        // Haversine 공식: 위경도 기반 지구상 거리 계산 (미터 단위)
        static double Haversine(ImageMetadata a, ImageMetadata b)
        {
            const double R = 6371000;
            double lat1 = a.position.latitude;
            double lon1 = a.position.longitude;
            double lat2 = b.position.latitude;
            double lon2 = b.position.longitude;

            double dLat = (lat2 - lat1) * Math.PI / 180;
            double dLon = (lon2 - lon1) * Math.PI / 180;
            double aa = Math.Sin(dLat / 2) * Math.Sin(dLat / 2) +
                       Math.Cos(lat1 * Math.PI / 180) * Math.Cos(lat2 * Math.PI / 180) *
                       Math.Sin(dLon / 2) * Math.Sin(dLon / 2);
            return R * 2 * Math.Atan2(Math.Sqrt(aa), Math.Sqrt(1 - aa));
        }
    }
}
