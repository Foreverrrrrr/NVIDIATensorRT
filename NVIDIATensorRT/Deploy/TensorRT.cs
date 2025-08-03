using NVIDIATensorRT.Custom;
using NVIDIATensorRT.SAHI;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using static NVIDIATensorRT.Custom.Nvinfer;

namespace NVIDIATensorRT.Deploy
{
    /// <summary>
    /// 调试日志级别
    /// </summary>
    public enum DebugLogLevel
    {
        /// <summary>
        /// 仅输出基本信息
        /// </summary>
        Basic = 0,
        /// <summary>
        /// 输出详细信息
        /// </summary>
        Info = 1,
        /// <summary>
        /// 输出详细的调试信息
        /// </summary>
        Detailed = 2,
        /// <summary>
        /// 输出所有调试信息（包括每个检测框的详细信息）
        /// </summary>
        Verbose = 3
    }

    /// <summary>
    /// TensorRT推理引擎
    /// </summary>
    /// <summary>
    /// TensorRT 推理封装：提供统一的同步/异步检测 API，并可选择启用 SAHI（切片大图）流程。
    /// - 非 SAHI：支持单图与批量（List&lt;Mat&gt;）一次推理，输出按原图坐标解析并做 NMS。
    /// - SAHI：自动将大图分块、打包推理，再将每块结果映射回全图并做全局 NMS。
    /// 线程关系：实例持有底层 <c>Nvinfer</c> 推理上下文，通常不要在多个线程并发使用同一实例以避免竞争。
    /// 
    /// 调试日志使用示例：
    /// <code>
    /// // 基础使用（避免重复输出）
    /// tensorRT.EnableInternalDebug();     // 启用内部调试输出
    /// tensorRT.SetDebugBasic();           // 设置为Basic级别 - 仅显示性能统计摘要
    /// tensorRT.SetDebugInfo();            // 设置为Info级别（默认）- 显示推理阶段信息
    /// tensorRT.SetDebugVerbose();         // 设置为Verbose级别 - 显示所有详细信息
    /// 
    /// // 日志订阅（无需添加额外前缀）
    /// tensorRT.NVIDIALogEvent += message => Console.WriteLine(message);
    /// </code>
    /// </summary>
    public unsafe class TensorRT
    {
        #region Fields & Types
        /// <summary>
        /// SAHI 分块的位置信息（相对于原始大图的 ROI 偏移与尺寸）
        /// </summary>
        public struct TileMeta
        {
            public int X;
            public int Y;
            public int W;
            public int H;
        }

        /// <summary>
        /// NVIDIA 推理引擎
        /// </summary>
        private readonly Nvinfer NVIDIAPredictor;

        /// <summary>
        /// TensorRT 相关日志事件
        /// </summary>
        public event Action<string> NVIDIALogEvent;

        /// <summary>
        /// 张量内存池
        /// </summary>
        private readonly ConcurrentQueue<float[]> MemoryPool = new ConcurrentQueue<float[]>();

        /// <summary>
        /// Mat 对象池（用于图像预处理优化）
        /// </summary>
        private readonly ConcurrentQueue<Mat> MatPool = new ConcurrentQueue<Mat>();

        /// <summary>
        /// 预处理参数缓存（避免重复计算缩放参数）
        /// </summary>
        private readonly ConcurrentDictionary<(int, int, int, int), (float scale, float padX, float padY)> PreprocessParamsCache
            = new ConcurrentDictionary<(int, int, int, int), (float, float, float)>();

        /// <summary>
        /// YAML 类别缓存
        /// </summary>
        public string[] YamlClass { get; private set; }

        /// <summary>
        /// YAML 缓存对应的来源路径，避免不同模型/数据集间错配
        /// </summary>
        private string _yamlClassPath;

        /// <summary>
        /// 启用解析调试日志（默认 false）。开启后在解析关键点输出调试信息。
        /// </summary>
        public bool DebugParsing { get; set; } = false;

        /// <summary>
        /// 调试日志级别（仅当 DebugParsing=true 时生效）
        /// </summary>
        public DebugLogLevel LogLevel { get; set; } = DebugLogLevel.Info;

        /// <summary>
        /// 每阶段最多打印的检测条目数（仅当 DebugParsing=true 时生效）。
        /// </summary>
        public int DebugMaxDetPrints { get; set; } = 3;

        // 便利方法：设置调试级别
        public void SetDebugLevel(DebugLogLevel level) => LogLevel = level;
        public void SetDebugBasic() => LogLevel = DebugLogLevel.Basic;       // 仅基础性能统计
        public void SetDebugInfo() => LogLevel = DebugLogLevel.Info;         // 包含推理阶段信息  
        public void SetDebugDetailed() => LogLevel = DebugLogLevel.Detailed; // 包含解析过程信息
        public void SetDebugVerbose() => LogLevel = DebugLogLevel.Verbose;   // 包含每个分块/NMS详情

        // 便利方法：控制内部调试输出（解决重复日志问题）
        public void EnableInternalDebug() => DebugParsing = true;
        public void DisableInternalDebug() => DebugParsing = false;

        // 性能分析：识别性能瓶颈
        public void EnablePerformanceAnalysis()
        {
            DebugParsing = true;
            LogLevel = DebugLogLevel.Basic; // 仅显示关键性能信息
        }

        /// <summary>
        /// 从内存池获取缓冲区
        /// </summary>
        private float[] GetBuffer(int size) => MemoryPool.TryDequeue(out var buf) && buf.Length >= size ? buf : new float[size];
        private void ReturnBuffer(float[] buf) => MemoryPool.Enqueue(buf);

        /// <summary>
        /// 从Mat池获取Mat对象，优化内存分配
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Mat GetMat(int height, int width, MatType type, Scalar fillValue = default)
        {
            if (MatPool.TryDequeue(out var mat) && mat.Width == width && mat.Height == height && mat.Type() == type && !mat.IsDisposed)
            {
                if (fillValue != default) mat.SetTo(fillValue);
                return mat;
            }
            return fillValue != default ? new Mat(height, width, type, fillValue) : new Mat(height, width, type);
        }

        /// <summary>
        /// 返还Mat对象到池中
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ReturnMat(Mat mat)
        {
            if (mat != null && !mat.IsDisposed && MatPool.Count < 50) // 限制池大小避免内存泄漏
            {
                MatPool.Enqueue(mat);
            }
            else
            {
                mat?.Dispose();
            }
        }

        /// <summary>
        /// 缓存预处理参数计算，避免重复计算
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private (float scale, float padX, float padY) CachedCalculatePreprocessParams(InputTensor inputTensor, OpenCvSharp.Size originalSize)
        {
            var key = (inputTensor.Width, inputTensor.Height, originalSize.Width, originalSize.Height);
            return PreprocessParamsCache.GetOrAdd(key, _ => CalculatePreprocessParams(inputTensor, originalSize));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void DebugLog(string message, DebugLogLevel level = DebugLogLevel.Info)
        {
            if (DebugParsing && LogLevel >= level)
                NVIDIALogEvent?.Invoke(message);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void DebugLog(string message)
        {
            if (DebugParsing)
                NVIDIALogEvent?.Invoke(message);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void VerboseLog(string message)
        {
            if (DebugParsing && LogLevel >= DebugLogLevel.Verbose)
                NVIDIALogEvent?.Invoke(message);
        }
        #endregion

        #region Constructors & Bindings

        /// <summary>
        /// 初始化空的 TensorRT 封装实例（稍后通过其它构造函数或绑定方法进行加载/配置）。
        /// </summary>
        public TensorRT()
        {

        }

        /// <summary>
        /// 加载 TensorRT 引擎文件（.engine）
        /// </summary>
        /// <param name="path_engine">TensorRT 引擎文件路径（.engine）。</param>
        /// <exception cref="FileNotFoundException">当文件不存在时抛出。</exception>
        public TensorRT(string path_engine)
        {
            if (File.Exists(path_engine))
            {
                NVIDIAPredictor = new Nvinfer(path_engine);
            }
            else
            {
                throw new FileNotFoundException($"指定的 TensorRT 引擎文件不存在: {path_engine}", path_engine);
            }
        }

        /// <summary>
        /// 加载 TensorRT 引擎文件并初始化输入张量
        /// </summary>
        /// <param name="path_engine">TensorRT 引擎文件路径（.engine）。</param>
        /// <param name="inputtensor">模型输入维度（绑定前用于设定 batch/CHW）。</param>
        /// <exception cref="FileNotFoundException">当文件不存在时抛出。</exception>
        public TensorRT(string path_engine, Dims inputtensor)
        {
            if (File.Exists(path_engine))
            {
                NVIDIAPredictor = new Nvinfer(path_engine, inputtensor.d[0]);
                NVIDIAPredictor.SetBindingDimensions(0, inputtensor);
            }
            else
            {
                throw new FileNotFoundException($"指定的 TensorRT 引擎文件不存在: {path_engine}", path_engine);
            }
        }

        /// <summary>
        /// 加载 TensorRT 引擎并获取输入输出张量结构
        /// </summary>
        /// <param name="path_engine">TensorRT 引擎文件路径（.engine）。</param>
        /// <param name="inputName">输入节点名称（绑定名）。</param>
        /// <param name="outputName">输出节点名称（绑定名）。</param>
        /// <param name="input">输出：输入张量维度信息。</param>
        /// <param name="output">输出：输出张量维度信息。</param>
        /// <exception cref="FileNotFoundException">当文件不存在时抛出。</exception>
        public TensorRT(string path_engine, string inputName, string outputName, ref InputTensor input, ref OutputTensor output)
        {
            if (!File.Exists(path_engine))
            {
                throw new FileNotFoundException($"指定的 TensorRT 引擎文件不存在: {path_engine}", path_engine);
            }
            NVIDIAPredictor = new Nvinfer(path_engine);
            Tensor(inputName, outputName, ref input, ref output);
        }

        /// <summary>
        /// 根据节点名称设置绑定的输入/输出张量的维度信息。
        /// </summary>
        /// <param name="nodeName">绑定名（输入或输出）。</param>
        /// <param name="dims">维度（CHW 或 BCHW）。</param>
        public void SetBindingDimensions(string nodeName, Dims dims)
        {
            NVIDIAPredictor.SetBindingDimensions(nodeName, dims);
        }

        /// <summary>
        /// 获取推理模型输入和输出张量的结构。
        /// </summary>
        /// <param name="inputstring">输入绑定名。</param>
        /// <param name="outputstring">输出绑定名。</param>
        /// <param name="input">输出：输入张量维度信息。</param>
        /// <param name="output">输出：输出张量维度信息。</param>
        public void Tensor(string inputstring, string outputstring, ref InputTensor input, ref OutputTensor output)
        {
            input = new InputTensor();
            output = new OutputTensor();
            Dims InputDims = NVIDIAPredictor.GetBindingDimensions(inputstring);
            input.BatchSize = InputDims.d[0];
            input.Channels = InputDims.d[1];
            input.Height = InputDims.d[2];
            input.Width = InputDims.d[3];
            InputDims = NVIDIAPredictor.GetBindingDimensions(outputstring);
            output.BatchSize = InputDims.d[0];
            output.Channels = InputDims.d[1];
            output.NumDetections = InputDims.d[2];
        }
        #endregion

        #region Unified API
        /// <summary>
        /// 统一的同步检测入口：可选启用 SAHI，返回原图坐标的检测结果（已做 NMS）。
        /// </summary>
        /// <param name="inputtensor">输入张量描述（包含目标尺寸等）。</param>
        /// <param name="outputtensor">输出张量描述（包含通道/检测数等）。</param>
        /// <param name="image">输入图像（BGR Mat）。</param>
        /// <param name="yamlpath">YOLO 数据集 .yaml 文件路径，用于加载类别名；可为空（则 ClassDescribe 为空）。</param>
        /// <param name="confidenceThreshold">置信度阈值。</param>
        /// <param name="iouThreshold">NMS IoU 阈值。</param>
        /// <param name="useSahi">是否启用 SAHI 大图分块模式。</param>
        /// <returns>按原图坐标系返回的检测结果列表。</returns>
        /// <exception cref="Exception">当底层推理器未初始化时。</exception>
        /// <remarks>
        /// 非 SAHI：单图将被打包为 1-batch 进行一次推理；坐标基于输入预处理（等比缩放+填充）逆变换还原。
        /// SAHI：转入 SAHI 流程，对大图切片后推理并合并结果（带全局 NMS）。
        /// </remarks>
        public unsafe List<DetectionResult> Detect(
            ref InputTensor inputtensor,
            ref OutputTensor outputtensor,
            Mat image,
            string yamlpath,
            float confidenceThreshold = 0.3f,
            float iouThreshold = 0.3f,
            bool useSahi = false)
        {
            DebugLog($"[Detect] 开始单图推理 - 输入尺寸: {image.Width}x{image.Height}, SAHI: {useSahi}, 置信度: {confidenceThreshold}, IoU: {iouThreshold}");

            if (useSahi)
            {
                DebugLog("[Detect] 使用SAHI模式进行大图推理");
                // 走已有的 SAHI 同步路径（预处理→推理→解析+全局 NMS）
                return SAHIInFerenceAndParse(
                    ref inputtensor,
                    ref outputtensor,
                    image,
                    yamlpath,
                    confidenceThreshold,
                    iouThreshold);
            }

            // 非 SAHI：单图同步推理 + 解析
            if (NVIDIAPredictor == null)
                throw new Exception("NVIDIAPredictor初始化失败！");

            DebugLog($"[Detect] 非SAHI模式 - 目标张量尺寸: {inputtensor.Width}x{inputtensor.Height}");

            var totalWatch = Stopwatch.StartNew();
            var sw = Stopwatch.StartNew();

            // 复用现有批处理，仅传 1 张
            DebugLog("[Detect] 开始图像预处理");
            var ptr = ProcessBatch(ref inputtensor, new List<Mat> { image });
            sw.Stop();
            long processBatchTime = sw.ElapsedMilliseconds;
            DebugLog($"[Detect] 图像预处理完成，耗时: {processBatchTime}ms, 数据长度: {ptr.Length}");

            sw.Restart();
            DebugLog("[Detect] 开始加载推理数据到GPU");
            NVIDIAPredictor.LoadInferenceData("images", ptr.DataPtr, ptr.Length);
            sw.Stop();
            long loadDataTime = sw.ElapsedMilliseconds;
            DebugLog($"[Detect] 数据加载完成，耗时: {loadDataTime}ms");

            sw.Restart();
            DebugLog("[Detect] 开始TensorRT推理");
            NVIDIAPredictor.Infer();
            sw.Stop();
            long inferTime = sw.ElapsedMilliseconds;
            DebugLog($"[Detect] TensorRT推理完成，耗时: {inferTime}ms");

            sw.Restart();
            DebugLog("[Detect] 开始获取推理结果 - GPU到CPU数据传输");
            IntPtr outPtr = NVIDIAPredictor.GetZeroCopyResult(ref outputtensor, "output0");
            ptr.Dispose();
            sw.Stop();
            long getResultTime = sw.ElapsedMilliseconds;

            // 添加传输性能诊断
            var dataSize = outputtensor.Channels * outputtensor.NumDetections * sizeof(float);
            var transferSpeedMBps = dataSize / 1024.0 / 1024.0 / (getResultTime / 1000.0);
            var diagLevel = getResultTime > 50 ? DebugLogLevel.Basic : DebugLogLevel.Info;
            DebugLog($"[Detect] 获取推理结果完成，耗时: {getResultTime}ms - 数据量: {dataSize / 1024:F1}KB, 传输速度: {transferSpeedMBps:F1}MB/s", diagLevel);

            // 解析（非 SAHI），坐标还原基于 inputtensor.InputImageSize[0]
            DebugLog($"[Detect] 开始解析YOLO输出 - 输出张量尺寸: {outputtensor.Channels}x{outputtensor.NumDetections}");
            var results = ParseYoloOutput((float*)outPtr, inputtensor, outputtensor, yamlpath, confidenceThreshold, iouThreshold);
            DebugLog($"[Detect] 解析完成，检测到 {results.Count} 个目标");

            totalWatch.Stop();
            long totalTime = totalWatch.ElapsedMilliseconds;
            DebugLog($"[Detect] 性能统计 - 预处理: {processBatchTime}ms, 加载: {loadDataTime}ms, 推理: {inferTime}ms, 结果: {getResultTime}ms, 总计: {totalTime}ms");
            return results;
        }

        /// <summary>
        /// 统一的同步检测入口（多图）：非 SAHI 走批量推理并按图解析；SAHI 情况逐图处理并返回逐图结果。
        /// </summary>
        /// <param name="inputtensor">输入张量描述（包含目标尺寸等）。</param>
        /// <param name="outputtensor">输出张量描述（包含通道/检测数等）。</param>
        /// <param name="images">多张 BGR Mat 图像。</param>
        /// <param name="yamlpath">YOLO 数据集 .yaml 文件路径，用于加载类别名；可为空。</param>
        /// <param name="confidenceThreshold">置信度阈值。</param>
        /// <param name="iouThreshold">NMS IoU 阈值。</param>
        /// <param name="useSahi">是否启用 SAHI 模式（逐图处理）。</param>
        /// <returns>逐图的检测结果集合，外层索引与输入 <paramref name="images"/> 对应。</returns>
        /// <exception cref="ArgumentException">当 <paramref name="images"/> 为空时。</exception>
        /// <exception cref="Exception">当底层推理器未初始化时。</exception>
        /// <remarks>
        /// 非 SAHI：批量打包为 NCHW，单次推理后按每张原图尺寸做坐标还原与按类 NMS。
        /// SAHI：每张图独立执行分块/推理/合并，返回逐图结果。
        /// </remarks>
        public unsafe List<List<DetectionResult>> Detect(
            ref InputTensor inputtensor,
            ref OutputTensor outputtensor,
            List<Mat> images,
            string yamlpath,
            float confidenceThreshold = 0.3f,
            float iouThreshold = 0.3f,
            bool useSahi = false)
        {
            if (images == null || images.Count == 0)
                throw new ArgumentException("images 不能为空");

            DebugLog($"[Detect-Batch] 开始批量推理 - 图像数量: {images.Count}, SAHI: {useSahi}, 置信度: {confidenceThreshold}, IoU: {iouThreshold}");

            if (useSahi)
            {
                DebugLog("[Detect-Batch] 使用SAHI模式逐图处理大图");
                // SAHI：逐图处理，保持行为一致（每张大图分块→推理→解析+合并）
                var all = new List<List<DetectionResult>>(images.Count);
                for (int i = 0; i < images.Count; i++)
                {
                    DebugLog($"[Detect-Batch] 处理第 {i + 1}/{images.Count} 张图像，尺寸: {images[i].Width}x{images[i].Height}");
                    var res = SAHIInFerenceAndParse(
                        ref inputtensor,
                        ref outputtensor,
                        images[i],
                        yamlpath,
                        confidenceThreshold,
                        iouThreshold);
                    DebugLog($"[Detect-Batch] 第 {i + 1} 张图像处理完成，检测到 {res.Count} 个目标");
                    all.Add(res);
                }
                DebugLog($"[Detect-Batch] SAHI批量处理完成，共处理 {all.Count} 张图像");
                return all;
            }

            if (NVIDIAPredictor == null)
                throw new Exception("NVIDIAPredictor初始化失败！");

            DebugLog($"[Detect-Batch] 非SAHI批量模式 - 目标张量尺寸: {inputtensor.Width}x{inputtensor.Height}");

            var totalWatch = Stopwatch.StartNew();
            var sw = Stopwatch.StartNew();

            // 非 SAHI：批量打包
            DebugLog("[Detect-Batch] 开始批量图像预处理");
            var safe = ProcessBatch(ref inputtensor, images);
            sw.Stop();
            long processBatchTime = sw.ElapsedMilliseconds;
            DebugLog($"[Detect-Batch] 批量预处理完成，耗时: {processBatchTime}ms, 批次大小: {images.Count}");

            sw.Restart();
            DebugLog("[Detect-Batch] 开始加载批量推理数据到GPU");
            NVIDIAPredictor.LoadInferenceData("images", safe.DataPtr, safe.Length);
            sw.Stop();
            long loadDataTime = sw.ElapsedMilliseconds;
            DebugLog($"[Detect-Batch] 批量数据加载完成，耗时: {loadDataTime}ms");

            sw.Restart();
            DebugLog("[Detect-Batch] 开始批量TensorRT推理");
            NVIDIAPredictor.Infer();
            sw.Stop();
            long inferTime = sw.ElapsedMilliseconds;
            DebugLog($"[Detect-Batch] 批量推理完成，耗时: {inferTime}ms");

            sw.Restart();
            DebugLog("[Detect-Batch] 开始获取批量推理结果");
            IntPtr outPtr = NVIDIAPredictor.GetZeroCopyResult(ref outputtensor, "output0");
            // 非 SAHI 批量解析：基于每张原图尺寸做坐标还原，按图做 NMS
            DebugLog($"[Detect-Batch] 开始解析批量YOLO输出 - 批次: {images.Count}, 输出张量: {outputtensor.Channels}x{outputtensor.NumDetections}");
            var parsed = SAHIBatchParseYoloOutput(
                (float*)outPtr,
                inputtensor,
                outputtensor,
                batchCount: images.Count,
                yamlpath: yamlpath,
                confidenceThreshold: confidenceThreshold,
                iouThreshold: iouThreshold);
            safe.Dispose();
            sw.Stop();
            long parseTime = sw.ElapsedMilliseconds;

            int totalDetections = parsed.Sum(batch => batch.Count);
            DebugLog($"[Detect-Batch] 批量解析完成，耗时: {parseTime}ms, 总检测数: {totalDetections}");

            totalWatch.Stop();
            long totalTime = totalWatch.ElapsedMilliseconds;
            DebugLog($"[Detect-Batch] 性能统计 - 预处理: {processBatchTime}ms, 加载: {loadDataTime}ms, 推理: {inferTime}ms, 解析: {parseTime}ms, 总计: {totalTime}ms");
            return parsed;
        }

        /// <summary>
        /// 统一的异步检测入口：可选启用 SAHI。回调收到 OutputTensor/Mat/hostData 后可在回调内自解析。
        /// </summary>
        /// <param name="inputtensor">输入张量描述。</param>
        /// <param name="outputtensor">输出张量描述（其中将附带解析所需元信息）。</param>
        /// <param name="image">输入图像（BGR Mat）。</param>
        /// <param name="yamlpath">YOLO 数据集 .yaml 文件路径。</param>
        /// <param name="confidenceThreshold">置信度阈值。</param>
        /// <param name="iouThreshold">NMS IoU 阈值。</param>
        /// <param name="callback">异步回调：签名 <c>CopyReasoningBack</c>，在回调内可使用 <paramref name="outputtensor"/> 提供的元信息自解析。</param>
        /// <param name="useSahi">是否启用 SAHI 模式。</param>
        /// <exception cref="Exception">当底层推理器未初始化时。</exception>
        /// <remarks>
        /// 非 SAHI：打包为 1-batch 推理，结果通过 <c>GetAsyncResult</c> 零拷贝派发。
        /// SAHI：仍采用分块推理，但解析交给回调端（元信息已填充进 <paramref name="outputtensor"/>）。
        /// </remarks>
        public unsafe void DetectAsync(
            ref InputTensor inputtensor,
            ref OutputTensor outputtensor,
            Mat image,
            string yamlpath,
            float confidenceThreshold,
            float iouThreshold,
            CopyReasoningBack callback,
            bool useSahi = false)
        {
            if (useSahi)
            {
                // 签名：SAHIInFerenceAndParse(ref in, ref out, Mat, string yaml, CopyReasoningBack, float conf=0.3, float iou=0.3)
                SAHIInFerenceAndParse(
                    ref inputtensor,
                    ref outputtensor,
                    image,
                    yamlpath,
                    callback,
                    confidenceThreshold,
                    iouThreshold);
                return;
            }

            if (NVIDIAPredictor == null)
                throw new Exception("NVIDIAPredictor初始化失败！");
            var totalWatch = Stopwatch.StartNew();
            var sw = Stopwatch.StartNew();
            var safe = ProcessBatch(ref inputtensor, new List<Mat> { image });
            sw.Stop();
            long processBatchTime = sw.ElapsedMilliseconds;
            sw.Restart();
            NVIDIAPredictor.LoadInferenceData("images", safe.DataPtr, safe.Length);
            sw.Stop();
            long loadDataTime = sw.ElapsedMilliseconds;
            sw.Restart();
            NVIDIAPredictor.Infer();
            sw.Stop();
            long inferTime = sw.ElapsedMilliseconds;
            outputtensor.InputWidth = inputtensor.Width;
            outputtensor.InputHeight = inputtensor.Height;
            outputtensor.InputImageSizeForRestore = new[] { image.Size() };
            outputtensor.Tiles = null; // 非 SAHI
            outputtensor.YamlPath = yamlpath;
            outputtensor.ConfThreshold = confidenceThreshold;
            outputtensor.IouThreshold = iouThreshold;
            sw.Restart();
            NVIDIAPredictor.GetAsyncResult("output0", outputtensor, image, callback, null);
            safe.Dispose();
            sw.Stop();
            long getResultTime = sw.ElapsedMilliseconds;
            totalWatch.Stop();
            long totalTime = totalWatch.ElapsedMilliseconds;
            DebugLog($"[DetectAsync] 性能统计 - 预处理: {processBatchTime}ms, 加载: {loadDataTime}ms, 推理: {inferTime}ms, 回调: {getResultTime}ms, 总计: {totalTime}ms");
        }

        /// <summary>
        /// 统一的异步检测入口（多图）：非 SAHI 走批量推理并一次性回调（回调内可按 batch 解析）；SAHI 情况逐图异步回调。
        /// </summary>
        /// <param name="inputtensor">输入张量描述。</param>
        /// <param name="outputtensor">输出张量描述（其中将附带解析所需元信息）。</param>
        /// <param name="images">多张 BGR Mat 图像。</param>
        /// <param name="yamlpath">YOLO 数据集 .yaml 文件路径。</param>
        /// <param name="confidenceThreshold">置信度阈值。</param>
        /// <param name="iouThreshold">NMS IoU 阈值。</param>
        /// <param name="callback">异步回调：用于接收推理输出指针与上下文元数据。</param>
        /// <param name="useSahi">是否启用 SAHI 模式（逐图回调）。</param>
        /// <exception cref="ArgumentException">当 <paramref name="images"/> 为空时。</exception>
        /// <exception cref="Exception">当底层推理器未初始化时。</exception>
        /// <remarks>
        /// 非 SAHI：批量一次推理与一次回调，<see cref="OutputTensor.InputImageSizeForRestore"/> 长度与输入张数一致；回调端可按 batch 逐图解析。
        /// SAHI：对每张图单独触发一次回调。
        /// </remarks>
        public unsafe void DetectAsync(
            ref InputTensor inputtensor,
            ref OutputTensor outputtensor,
            List<Mat> images,
            string yamlpath,
            float confidenceThreshold,
            float iouThreshold,
            CopyReasoningBack callback,
            bool useSahi = false)
        {
            if (images == null || images.Count == 0)
                throw new ArgumentException("images 不能为空");

            if (useSahi)
            {
                // SAHI：逐图异步派发，便于外部在每次回调内解析对应大图
                foreach (var img in images)
                {
                    SAHIInFerenceAndParse(
                        ref inputtensor,
                        ref outputtensor,
                        img,
                        yamlpath,
                        callback,
                        confidenceThreshold,
                        iouThreshold);
                }
                return;
            }

            if (NVIDIAPredictor == null)
                throw new Exception("NVIDIAPredictor初始化失败！");

            var totalWatch = Stopwatch.StartNew();
            var sw = Stopwatch.StartNew();

            // 非 SAHI：批量打包
            var safe = ProcessBatch(ref inputtensor, images);
            sw.Stop();
            long processBatchTime = sw.ElapsedMilliseconds;

            sw.Restart();
            NVIDIAPredictor.LoadInferenceData("images", safe.DataPtr, safe.Length);
            sw.Stop();
            long loadDataTime = sw.ElapsedMilliseconds;

            sw.Restart();
            NVIDIAPredictor.Infer();
            sw.Stop();
            long inferTime = sw.ElapsedMilliseconds;

            // 在派发前，填充批量解析所需元信息
            outputtensor.InputWidth = inputtensor.Width;
            outputtensor.InputHeight = inputtensor.Height;
            outputtensor.InputImageSizeForRestore = inputtensor.InputImageSize; // 每张原图尺寸
            outputtensor.Tiles = null; // 非 SAHI
            outputtensor.YamlPath = yamlpath;
            outputtensor.ConfThreshold = confidenceThreshold;
            outputtensor.IouThreshold = iouThreshold;

            sw.Restart();
            // 与单图一致：零拷贝 + 回调。此处传入首图，仅作为上下文（若回调未使用可忽略）。
            NVIDIAPredictor.GetAsyncResult("output0", outputtensor, images[0], callback, null);
            safe.Dispose();
            sw.Stop();
            long getResultTime = sw.ElapsedMilliseconds;

            totalWatch.Stop();
            long totalTime = totalWatch.ElapsedMilliseconds;
            DebugLog($"[DetectAsync-Batch] 性能统计 - 预处理: {processBatchTime}ms, 加载: {loadDataTime}ms, 推理: {inferTime}ms, 回调: {getResultTime}ms, 总计: {totalTime}ms");
        }
        #endregion

        #region SAHI 超图推理（分块/解析/合并）
        /// <summary>
        /// SAHI 预处理（返回分块元信息），便于推理后做全图坐标还原与融合。
        /// </summary>
        /// <param name="inputtensor">输入张量描述（目标尺寸、通道等）。</param>
        /// <param name="images">大图 Mat（将进行分块）。</param>
        /// <param name="tiles">输出：每个切片在原图中的 ROI 信息（X、Y、W、H）。</param>
        /// <returns>封装了批量数据的 <c>SAHISafeTensor</c>，需在使用完毕后 <c>Dispose()</c>。</returns>
        /// <exception cref="Exception">当输入图像无效或内部预处理失败时。</exception>
        private unsafe SAHISafeTensor SAHIProcessBatchWithMeta(ref InputTensor inputtensor, Mat images, out List<TileMeta> tiles)
        {
            DebugLog($"[SAHIProcessBatch] 开始SAHI预处理 - 输入图像: {images.Width}x{images.Height}");

            Stopwatch swTotal = Stopwatch.StartNew();
            Stopwatch sw = new Stopwatch();
            if (images.Empty())
                throw new Exception("输入images错误");

            sw.Restart();
            DebugLog($"[SAHIProcessBatch] 开始图像分块 - 目标分块尺寸: {inputtensor.Width}x{inputtensor.Height}");
            var Partition = NVIDIATensorRT.SAHI.SAHI.SupergraphPartitionFixed(
                inputtensor.Height,
                inputtensor.Width,
                images
            );
            sw.Stop();
            var timePartition = sw.ElapsedMilliseconds;
            DebugLog($"[SAHIProcessBatch] 分块完成 - 分块数量: {Partition.Count}, 耗时: {timePartition}ms");

            tiles = ExtractTileMetas(Partition);
            int actualBatch = Partition.Count;
            int singleImageSize = inputtensor.Channels * inputtensor.Height * inputtensor.Width;
            DebugLog($"[SAHIProcessBatch] 分块元信息 - 批次: {actualBatch}, 单块大小: {singleImageSize}");

            float[] batchData = GetBuffer(actualBatch * singleImageSize);
            try
            {
                OpenCvSharp.Size[] InputImageSize = new OpenCvSharp.Size[actualBatch];
                int Width = inputtensor.Width;
                int Height = inputtensor.Height;
                int Channels = inputtensor.Channels;

                sw.Restart();
                var handle = GCHandle.Alloc(batchData, GCHandleType.Pinned);
                try
                {
                    IntPtr basePtr = handle.AddrOfPinnedObject();
                    var tileMetas = tiles;
                    VerboseLog("[SAHIProcessBatch] 开始并行处理分块");
                    Parallel.For(0, actualBatch, i =>
                    {
                        InputImageSize[i] = new OpenCvSharp.Size(tileMetas[i].W, tileMetas[i].H);
                        if (DebugParsing && i < DebugMaxDetPrints)
                            VerboseLog($"[SAHIProcessBatch] 处理分块 {i}: {tileMetas[i].W}x{tileMetas[i].H} at ({tileMetas[i].X},{tileMetas[i].Y})");
                        using (var processed = OptimizedPreprocessImage(Partition[i].Image, Width, Height))
                        {
                            int offset = i * singleImageSize;
                            if (processed.Total() != singleImageSize)
                                throw new InvalidOperationException($"图像大小不匹配 {singleImageSize}, got {processed.Total()}");
                            unsafe
                            {
                                fixed (float* pDest = &batchData[offset])
                                {
                                    Buffer.MemoryCopy(processed.DataPointer, pDest, processed.Total() * sizeof(float), processed.Total() * sizeof(float));
                                }
                            }
                        }
                    });
                }
                finally
                {
                    if (handle.IsAllocated) handle.Free();
                }
                sw.Stop();
                var timePreprocessAndPack = sw.ElapsedMilliseconds;

                sw.Restart();
                inputtensor.InputImageSize = InputImageSize;
                var tensor = new SAHISafeTensor(batchData, new[] { actualBatch, inputtensor.Channels, Height, Width }, Partition);
                sw.Stop();
                var timeTensor = sw.ElapsedMilliseconds;

                swTotal.Stop();
                DebugLog($"[SAHIProcessBatchWithMeta] 耗时统计 - 分块: {timePartition}ms, 预处理: {timePreprocessAndPack}ms, 张量: {timeTensor}ms, 总计: {swTotal.ElapsedMilliseconds}ms");
                return tensor;
            }
            catch (Exception)
            {
                ReturnBuffer(batchData);
                throw;
            }
        }

        /// <summary>
        /// 通过反射从 SAHI 分块对象集合中尽力抽取 (X,Y,W,H) 元信息。
        /// </summary>
        /// <param name="partition">SAHI 分块集合，元素应包含 <c>Rect</c>/<c>ROI</c> 或 <c>Image</c> 等字段/属性。</param>
        /// <returns>与分块顺序一致的 <see cref="TileMeta"/> 列表。</returns>
        private static List<TileMeta> ExtractTileMetas(System.Collections.IList partition)
        {
            var tiles = new List<TileMeta>(partition.Count);
            for (int i = 0; i < partition.Count; i++)
            {
                var p = partition[i];
                var t = p.GetType();
                int x = 0, y = 0, w = 0, h = 0;

                // 先尝试从 Rect/Roi 类型属性获取
                var rectProp =
                    GetPropCI(t, "Rect") ?? GetPropCI(t, "ROI") ?? GetPropCI(t, "Roi") ?? GetPropCI(t, "CropRect");
                var rectObj = rectProp?.GetValue(p);
                if (rectObj != null)
                {
                    if (rectObj is OpenCvSharp.Rect r)
                    {
                        x = r.X; y = r.Y; w = r.Width; h = r.Height;
                    }
                    else if (rectObj is System.Drawing.Rectangle dr)
                    {
                        x = dr.X; y = dr.Y; w = dr.Width; h = dr.Height;
                    }
                }

                // 若存在 SliceMat.Peak 或类似 TopLeft，使用其作为原图中的左上角偏移
                var peakProp = GetPropCI(t, "Peak") ?? GetPropCI(t, "TopLeft") ?? GetPropCI(t, "Origin");
                var peakObj = peakProp?.GetValue(p);
                if (peakObj != null)
                {
                    if (peakObj is OpenCvSharp.Point pt)
                    {
                        x = pt.X; y = pt.Y;
                    }
                    else if (peakObj is System.Drawing.Point dpt)
                    {
                        x = dpt.X; y = dpt.Y;
                    }
                }

                if (rectObj == null && peakObj == null)
                {
                    // 尝试 X/Y/OffsetX/OffsetY
                    x = TryGetMemberInt(t, p, new[] { "X", "OffsetX", "XStart", "StartX", "Left" }) ?? 0;
                    y = TryGetMemberInt(t, p, new[] { "Y", "OffsetY", "YStart", "StartY", "Top" }) ?? 0;
                    w = TryGetMemberInt(t, p, new[] { "Width", "W" }) ?? 0;
                    h = TryGetMemberInt(t, p, new[] { "Height", "H" }) ?? 0;
                }

                if (w == 0 || h == 0)
                {
                    // 回退到 Image 尺寸
                    var img = (GetPropCI(t, "Image")?.GetValue(p) ?? GetFieldCI(t, "Image")?.GetValue(p)) as Mat;
                    if (img != null)
                    {
                        w = img.Width; h = img.Height;
                    }
                }

                tiles.Add(new TileMeta { X = x, Y = y, W = w, H = h });
            }
            return tiles;
        }

        /// <summary>
        /// 反射获取不区分大小写的公共/非公共实例属性信息。
        /// </summary>
        /// <param name="t">类型。</param>
        /// <param name="name">属性名。</param>
        /// <returns>属性信息，若不存在则为 null。</returns>
        private static System.Reflection.PropertyInfo GetPropCI(Type t, string name)
        {
            return t.GetProperty(name, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.IgnoreCase);
        }

        /// <summary>
        /// 反射获取不区分大小写的公共/非公共实例字段信息。
        /// </summary>
        /// <param name="t">类型。</param>
        /// <param name="name">字段名。</param>
        /// <returns>字段信息，若不存在则为 null。</returns>
        private static System.Reflection.FieldInfo GetFieldCI(Type t, string name)
        {
            return t.GetField(name, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.IgnoreCase);
        }

        /// <summary>
        /// 从给定字段/属性名集合中尝试读取整型值（大小写不敏感）。
        /// </summary>
        /// <param name="t">类型。</param>
        /// <param name="instance">对象实例。</param>
        /// <param name="names">候选成员名。</param>
        /// <returns>读取成功则返回整型值，否则为 null。</returns>
        private static int? TryGetMemberInt(Type t, object instance, string[] names)
        {
            foreach (var n in names)
            {
                var pi = GetPropCI(t, n);
                if (pi != null)
                {
                    var v = pi.GetValue(instance);
                    if (v != null)
                    {
                        try { return Convert.ToInt32(v); } catch { }
                    }
                }
                var fi = GetFieldCI(t, n);
                if (fi != null)
                {
                    var v = fi.GetValue(instance);
                    if (v != null)
                    {
                        try { return Convert.ToInt32(v); } catch { }
                    }
                }
            }
            return null;
        }

        /// <summary>
        /// SAHI 推理并解析为原图坐标的检测结果（同步、合并所有分块，带全局 NMS）。
        /// </summary>
        /// <param name="inputtensor">输入张量描述。</param>
        /// <param name="outputtensor">输出张量描述。</param>
        /// <param name="largeImage">大图 Mat。</param>
        /// <param name="yamlpath">YOLO yaml 路径（类别名）。</param>
        /// <param name="confidenceThreshold">置信度阈值。</param>
        /// <param name="iouThreshold">NMS IoU 阈值。</param>
        /// <returns>映射至原图坐标并做全局 NMS 的检测结果。</returns>
        /// <exception cref="Exception">当底层推理器未初始化时。</exception>
        /// <remarks>已标记为 Obsolete，推荐使用 Detect(useSahi: true)。</remarks>
        [Obsolete("请改用 Detect(useSahi: true) 2025.9.2弃用")]
        public unsafe List<DetectionResult> SAHIInFerenceAndParse(
                ref InputTensor inputtensor,
                ref OutputTensor outputtensor,
                Mat largeImage,
                string yamlpath,
                float confidenceThreshold = 0.3f,
                float iouThreshold = 0.3f)
        {
            if (NVIDIAPredictor == null)
                throw new Exception("NVIDIAPredictor初始化失败！");
            var originalSize = largeImage.Size();
            Stopwatch totalWatch = Stopwatch.StartNew();
            Stopwatch sw = Stopwatch.StartNew();
            var tensor = SAHIProcessBatchWithMeta(ref inputtensor, largeImage, out var tiles);
            sw.Stop();
            long processBatchTime = sw.ElapsedMilliseconds;
            sw.Restart();
            NVIDIAPredictor.LoadInferenceData("images", tensor.DataPtr, tensor.Length);
            sw.Stop();
            long loadDataTime = sw.ElapsedMilliseconds;
            sw.Restart();
            NVIDIAPredictor.Infer();
            sw.Stop();
            long inferTime = sw.ElapsedMilliseconds;
            sw.Restart();
            IntPtr outPtr = NVIDIAPredictor.GetZeroCopyResult(ref outputtensor, "output0");
            var perTile = SAHIBatchParseYoloOutputUsingTiles(
                (float*)outPtr,
                inputtensor,
                outputtensor,
                tiles,
                yamlpath,
                confidenceThreshold,
                iouThreshold);
            tensor.Dispose();
            var classNames = GetClassNamesFromYaml(yamlpath);
            var merged = MergeSahiDetections(perTile, tiles, originalSize, classNames, iouThreshold);
            sw.Stop();
            long parseTime = sw.ElapsedMilliseconds;
            totalWatch.Stop();
            long totalTime = totalWatch.ElapsedMilliseconds;
            DebugLog($"[SAHI] 性能统计 - 预处理: {processBatchTime}ms, 加载: {loadDataTime}ms, 推理: {inferTime}ms, 解析: {parseTime}ms, 总计: {totalTime}ms");

            // 添加完整的SAHI统计摘要
            var tileCount = tiles?.Count ?? 0;
            var imageSizeStr = $"{largeImage.Width}x{largeImage.Height}";
            var tileSize = $"{inputtensor.Width}x{inputtensor.Height}";
            var resultCount = merged?.Count ?? 0;
            DebugLog($"[SAHI-完整统计] 原图: {imageSizeStr}, 分块: {tileCount}个({tileSize}), 检测结果: {resultCount}个, 总耗时: {totalTime}ms", DebugLogLevel.Basic);
            return merged;
        }

        /// <summary>
        /// SAHI 推理（异步回调版）：使用异步回调获取输出，供外部在回调中自行处理结果。
        /// </summary>
        /// <param name="inputtensor">输入张量描述。</param>
        /// <param name="outputtensor">输出张量描述（派发前会填充 SAHI 解析所需元信息）。</param>
        /// <param name="largeImage">大图 Mat。</param>
        /// <param name="yamlpath">YOLO yaml 路径。</param>
        /// <param name="Actioncopy">回调函数。</param>
        /// <param name="confidenceThreshold">置信度阈值。</param>
        /// <param name="iouThreshold">NMS IoU 阈值。</param>
        /// <exception cref="Exception">当底层推理器未初始化时。</exception>
        /// <remarks>已标记为 Obsolete，推荐使用 DetectAsync(useSahi: true)。</remarks>
        [Obsolete("请改用 DetectAsync(useSahi: true)，并在回调内解析 2025.9.2弃用")]
        public unsafe void SAHIInFerenceAndParse(
                ref InputTensor inputtensor,
                ref OutputTensor outputtensor,
                Mat largeImage,
                string yamlpath,
                CopyReasoningBack Actioncopy,
                float confidenceThreshold = 0.3f,
                float iouThreshold = 0.3f)
        {
            if (NVIDIAPredictor == null)
                throw new Exception("NVIDIAPredictor初始化失败！");

            Stopwatch totalWatch = Stopwatch.StartNew();
            Stopwatch sw = Stopwatch.StartNew();
            // 仍然进行 SAHI 预处理（含分块），但解析交由回调处理
            var tensor = SAHIProcessBatchWithMeta(ref inputtensor, largeImage, out var tiles);
            sw.Stop();
            long processBatchTime = sw.ElapsedMilliseconds;

            sw.Restart();
            NVIDIAPredictor.LoadInferenceData("images", tensor.DataPtr, tensor.Length);
            sw.Stop();
            long loadDataTime = sw.ElapsedMilliseconds;

            sw.Restart();
            NVIDIAPredictor.Infer();
            sw.Stop();
            long inferTime = sw.ElapsedMilliseconds;

            sw.Restart();
            // 在派发回调前，将解析所需的元信息打包进 outputtensor，便于回调内独立解析
            outputtensor.InputWidth = inputtensor.Width;
            outputtensor.InputHeight = inputtensor.Height;
            outputtensor.InputImageSizeForRestore = inputtensor.InputImageSize; // 每个 tile 的 ROI 尺寸
            outputtensor.Tiles = tiles; // SAHI 偏移/尺寸
            outputtensor.YamlPath = yamlpath;
            outputtensor.ConfThreshold = confidenceThreshold;
            outputtensor.IouThreshold = iouThreshold;

            // 将结果异步返回给调用方提供的回调
            NVIDIAPredictor.GetAsyncResult("output0", outputtensor, largeImage, Actioncopy, null);
            tensor.Dispose();
            sw.Stop();
            long getResultTime = sw.ElapsedMilliseconds;
            totalWatch.Stop();
            long totalTime = totalWatch.ElapsedMilliseconds;
            DebugLog($"[SAHI-Async] 性能统计 - 预处理: {processBatchTime}ms, 加载: {loadDataTime}ms, 推理: {inferTime}ms, 回调: {getResultTime}ms, 总计: {totalTime}ms");

            // 添加完整的SAHI统计摘要
            var tileCount = tiles?.Count ?? 0;
            var imageSizeStr = $"{largeImage.Width}x{largeImage.Height}";
            var tileSize = $"{inputtensor.Width}x{inputtensor.Height}";
            DebugLog($"[SAHI-完整统计] 原图: {imageSizeStr}, 分块: {tileCount}个({tileSize}), 总耗时: {totalTime}ms (分块+预处理: {processBatchTime}ms, 推理: {inferTime}ms)", DebugLogLevel.Basic);
        }

        /// <summary>
        /// 执行一次 SAHI 推理（旧版 Obsolete）。
        /// </summary>
        /// <param name="inputtensor">输入张量描述。</param>
        /// <param name="outputtensor">输出张量描述。</param>
        /// <param name="values">大图 Mat。</param>
        /// <param name="Actioncopy">回调函数。</param>
        /// <exception cref="Exception">当底层推理器未初始化时。</exception>
        /// <remarks>已标记为 Obsolete，推荐使用 DetectAsync(useSahi: true)。</remarks>
        [Obsolete("请改用 DetectAsync(useSahi: true) 2025.9.2弃用")]
        public void SAHIInFerencePtr(ref InputTensor inputtensor, ref OutputTensor outputtensor, Mat values, CopyReasoningBack Actioncopy)
        {
            if (NVIDIAPredictor == null)
                throw new Exception("NVIDIAPredictor初始化失败！");
            Stopwatch totalWatch = Stopwatch.StartNew();
            Stopwatch sw = Stopwatch.StartNew();
            // 使用带 meta 的 SAHI 预处理，便于回调内解析
            var intptr = SAHIProcessBatchWithMeta(ref inputtensor, values, out var tiles);
            sw.Stop();
            long processBatchTime = sw.ElapsedMilliseconds;
            sw.Restart();
            NVIDIAPredictor.LoadInferenceData("images", intptr.DataPtr, intptr.Length);
            sw.Stop();
            long loadDataTime = sw.ElapsedMilliseconds;
            sw.Restart();
            NVIDIAPredictor.Infer();
            sw.Stop();
            long inferTime = sw.ElapsedMilliseconds;
            sw.Restart();
            // 在派发回调前，填充解析所需元数据
            outputtensor.InputWidth = inputtensor.Width;
            outputtensor.InputHeight = inputtensor.Height;
            outputtensor.InputImageSizeForRestore = inputtensor.InputImageSize;
            outputtensor.Tiles = tiles;
            outputtensor.ConfThreshold = 0.3f;
            outputtensor.IouThreshold = 0.3f;
            NVIDIAPredictor.GetAsyncResult("output0", outputtensor, values, Actioncopy, null);
            intptr.Dispose();
            sw.Stop();
            long getResultTime = sw.ElapsedMilliseconds;
            totalWatch.Stop();
            long totalTime = totalWatch.ElapsedMilliseconds;
            DebugLog($"[ProcessBatch-1] 性能统计 - 预处理: {processBatchTime}ms, 加载: {loadDataTime}ms, 推理: {inferTime}ms, 结果: {getResultTime}ms, 总计: {totalTime}ms");
        }

        /// <summary>
        /// SAHI 批量解析（局部坐标）：基于每个 tile 的原始 ROI 尺寸计算 scale/pad，并还原到 tile 局部坐标。
        /// </summary>
        /// <param name="output">模型输出指针（按 [B, C, N] 展平）。</param>
        /// <param name="inshape">输入张量描述。</param>
        /// <param name="outshape">输出张量描述。</param>
        /// <param name="tiles">tile 元信息（用于确定 ROI 尺寸）。</param>
        /// <param name="yamlpath">YOLO yaml 路径。</param>
        /// <param name="confidenceThreshold">置信度阈值。</param>
        /// <param name="iouThreshold">NMS IoU 阈值。</param>
        /// <returns>每个 tile 的检测结果（已在各自 tile 内按类 NMS）。</returns>
        /// <exception cref="ArgumentNullException">当 <paramref name="output"/> 为 null 时。</exception>
        /// <exception cref="ArgumentException">当 <paramref name="tiles"/> 为空时。</exception>
        private unsafe List<List<DetectionResult>> SAHIBatchParseYoloOutputUsingTiles(
            float* output,
            InputTensor inshape,
            OutputTensor outshape,
            List<TileMeta> tiles,
            string yamlpath,
            float confidenceThreshold = 0.3f,
            float iouThreshold = 0.3f)
        {
            if (output == null) throw new ArgumentNullException(nameof(output), "输出数据不能为空！");
            if (tiles == null || tiles.Count == 0) throw new ArgumentException("tiles 不能为空");

            int batchSize = tiles.Count;
            int channels = outshape.Channels;
            int numDetections = outshape.NumDetections;
            if (channels < 5) throw new ArgumentException($"通道大小错误：{channels}！");
            int numClasses = channels - 4; // 固定布局：x,y,w,h,cls...
            DebugLog($"[SAHIParseUsingTiles] 开始解析分块结果 - 分块数: {batchSize}, 通道数: {channels}, 检测数: {numDetections}, 类别数: {numClasses}");
            var perBatch = new List<DetectionResult>[batchSize];

            Parallel.For(0, batchSize, b =>
            {
                var roiSize = new OpenCvSharp.Size(tiles[b].W, tiles[b].H);
                (float scale, float padX, float padY) = CachedCalculatePreprocessParams(inshape, roiSize);
                if (DebugParsing && b < DebugMaxDetPrints)
                    DebugLog($"[SAHIParseUsingTiles] 分块{b}: ROI={roiSize.Width}x{roiSize.Height}, 缩放={scale:F5}, 填充=({padX:F2},{padY:F2})");

                var results = new List<DetectionResult>(numDetections);
                int baseBatch = b * channels * numDetections;
                float* basePtr = output + baseBatch;
                for (int i = 0; i < numDetections; i++)
                {
                    float cx = *(basePtr + 0 * numDetections + i);
                    float cy = *(basePtr + 1 * numDetections + i);
                    float w = *(basePtr + 2 * numDetections + i);
                    float h = *(basePtr + 3 * numDetections + i);
                    bool normalized = (cx <= 1 && cy <= 1 && w <= 1 && h <= 1);
                    if (normalized)
                    {
                        cx *= inshape.Width; cy *= inshape.Height; w *= inshape.Width; h *= inshape.Height;
                    }
                    int bestClassId = -1;
                    float maxClassScore = float.MinValue;
                    for (int c = 0; c < numClasses; c++)
                    {
                        float score = *(basePtr + (4 + c) * numDetections + i);
                        if (score > maxClassScore)
                        {
                            maxClassScore = score;
                            bestClassId = c;
                        }
                    }
                    if (maxClassScore < confidenceThreshold)
                        continue;

                    RectF rect = ConvertToOriginalCoords(cx, cy, w, h, scale, padX, padY, roiSize);
                    rect = ClampRect(rect, roiSize.Width, roiSize.Height);
                    if (DebugParsing && b < DebugMaxDetPrints && i < DebugMaxDetPrints)
                        DebugLog($"[SAHIParseUsingTiles] 分块{b}检测{i}: 类别={bestClassId}, 置信度={maxClassScore:F4}, 框=({rect.X:F1},{rect.Y:F1},{rect.Width:F1},{rect.Height:F1})");
                    results.Add(new DetectionResult
                    {
                        BatchId = b,
                        ClassId = bestClassId,
                        Confidence = maxClassScore,
                        BoundingBox = rect
                    });
                }
                perBatch[b] = results;
            });

            var classdescribe = GetClassNamesFromYaml(yamlpath);
            var finalResults = new List<List<DetectionResult>>(batchSize);
            for (int b = 0; b < batchSize; b++)
            {
                var result = ApplyClassAwareNMS(perBatch[b], classdescribe, iouThreshold);
                finalResults.Add(result);
                if (DebugParsing)
                    DebugLog($"[SAHIParseUsingTiles] 分块{b}NMS结果: 输入={perBatch[b].Count}, 输出={result.Count}");
            }
            return finalResults;
        }

        /// <summary>
        /// SAHI 批量解析（指定 batchCount），避免依赖 <see cref="OutputTensor.BatchSize"/>。
        /// </summary>
        /// <param name="output">输出指针。</param>
        /// <param name="inshape">输入张量描述。</param>
        /// <param name="outshape">输出张量描述。</param>
        /// <param name="batchCount">批量数量。</param>
        /// <param name="yamlpath">YOLO yaml 路径。</param>
        /// <param name="confidenceThreshold">置信度阈值。</param>
        /// <param name="iouThreshold">NMS IoU 阈值。</param>
        /// <returns>按 batch 的检测结果集合。</returns>
        /// <exception cref="ArgumentNullException">当 <paramref name="output"/> 为 null 时。</exception>
        private unsafe List<List<DetectionResult>> SAHIBatchParseYoloOutput(
            float* output,
            InputTensor inshape,
            OutputTensor outshape,
            int batchCount,
            string yamlpath,
            float confidenceThreshold = 0.3f,
            float iouThreshold = 0.3f)
        {
            if (output == null) throw new ArgumentNullException(nameof(output), "输出数据不能为空！");
            int channels = outshape.Channels;
            int numDetections = outshape.NumDetections;
            if (channels < 5) throw new ArgumentException($"通道大小错误：{channels}！");
            int numClasses = channels - 4; // 固定布局：x,y,w,h,cls...
            DebugLog($"[SAHIBatchParse] 开始批量解析SAHI结果 - 批次: {batchCount}, 通道数: {channels}, 检测数: {numDetections}, 类别数: {numClasses}");

            var perBatch = new List<DetectionResult>[batchCount];
            Parallel.For(0, batchCount, b =>
            {
                (float scale, float padX, float padY) = CachedCalculatePreprocessParams(inshape, inshape.InputImageSize[b]);
                var results = new List<DetectionResult>(numDetections);
                int baseBatch = b * channels * numDetections;
                float* basePtr = output + baseBatch;
                for (int i = 0; i < numDetections; i++)
                {
                    float cx = *(basePtr + 0 * numDetections + i);
                    float cy = *(basePtr + 1 * numDetections + i);
                    float w = *(basePtr + 2 * numDetections + i);
                    float h = *(basePtr + 3 * numDetections + i);
                    bool normalized = (cx <= 1 && cy <= 1 && w <= 1 && h <= 1);
                    if (normalized)
                    {
                        cx *= inshape.Width; cy *= inshape.Height; w *= inshape.Width; h *= inshape.Height;
                    }
                    int bestClassId = -1;
                    float maxClassScore = float.MinValue;
                    for (int c = 0; c < numClasses; c++)
                    {
                        float score = *(basePtr + (4 + c) * numDetections + i);
                        if (score > maxClassScore)
                        {
                            maxClassScore = score;
                            bestClassId = c;
                        }
                    }
                    if (maxClassScore < confidenceThreshold)
                        continue;
                    RectF rect = ConvertToOriginalCoords(cx, cy, w, h, scale, padX, padY, inshape.InputImageSize[b]);
                    rect = ClampRect(rect, inshape.InputImageSize[b].Width, inshape.InputImageSize[b].Height);
                    results.Add(new DetectionResult { BatchId = b, ClassId = bestClassId, Confidence = maxClassScore, BoundingBox = rect });
                }
                perBatch[b] = results;
            });
            var classdescribe = GetClassNamesFromYaml(yamlpath);
            var finalResults = new List<List<DetectionResult>>(batchCount);
            for (int b = 0; b < batchCount; b++)
            {
                var result = ApplyClassAwareNMS(perBatch[b], classdescribe, iouThreshold);
                finalResults.Add(result);
                if (DebugParsing)
                    VerboseLog($"[SAHIBatchParse] 批次{b}NMS结果: 输入={perBatch[b].Count}, 输出={result.Count}");
            }
            return finalResults;
        }

        /// <summary>
        /// SAHI 批量解析（全局坐标）并做一次全局 NMS。
        /// </summary>
        /// <param name="output">输出指针。</param>
        /// <param name="inshape">输入张量描述。</param>
        /// <param name="outshape">输出张量描述。</param>
        /// <param name="tiles">切片元信息。</param>
        /// <param name="originalSize">原始全图尺寸。</param>
        /// <param name="yamlpath">YOLO yaml 路径。</param>
        /// <param name="confidenceThreshold">置信度阈值。</param>
        /// <param name="iouThreshold">全局 NMS IoU 阈值。</param>
        /// <returns>映射到全图坐标并经全局 NMS 的检测结果。</returns>
        public unsafe List<DetectionResult> SAHIBatchParseYoloOutput(
            float* output,
            ref InputTensor inshape,
            ref OutputTensor outshape,
            List<TileMeta> tiles,
            OpenCvSharp.Size originalSize,
            string yamlpath,
            float confidenceThreshold = 0.3f,
            float iouThreshold = 0.3f)
        {
            if (tiles == null || tiles.Count == 0)
                return new List<DetectionResult>();

            // 先解析为 tile 局部坐标
            var perTile = SAHIBatchParseYoloOutput(
                output,
                inshape,
                outshape,
                batchCount: tiles.Count,
                yamlpath: yamlpath,
                confidenceThreshold: confidenceThreshold,
                iouThreshold: iouThreshold);

            // 再叠加偏移并全局 NMS
            var classNames = GetClassNamesFromYaml(yamlpath);
            return MergeSahiDetections(perTile, tiles, originalSize, classNames, iouThreshold);
        }

        /// <summary>
        /// 将每个 tile 的检测（已在 tile 内做 NMS）映射到原图坐标并做一次全局 NMS。
        /// </summary>
        /// <param name="perTile">逐 tile 的检测结果集。</param>
        /// <param name="tiles">tile 元信息列表。</param>
        /// <param name="originalSize">原图尺寸。</param>
        /// <param name="classNames">类别名数组。</param>
        /// <param name="iouThreshold">全局 NMS IoU 阈值。</param>
        /// <returns>全局融合后的检测结果。</returns>
        private List<DetectionResult> MergeSahiDetections(
            List<List<DetectionResult>> perTile,
            List<TileMeta> tiles,
            OpenCvSharp.Size originalSize,
            string[] classNames,
            float iouThreshold)
        {
            var all = new List<DetectionResult>();
            int n = Math.Min(perTile?.Count ?? 0, tiles?.Count ?? 0);
            for (int b = 0; b < n; b++)
            {
                var tileRes = perTile[b];
                var meta = tiles[b];
                if (tileRes == null) continue;
                foreach (var det in tileRes)
                {
                    var box = det.BoundingBox;
                    var mapped = new RectF(
                        box.X + meta.X,
                        box.Y + meta.Y,
                        box.Width,
                        box.Height);
                    mapped = ClampRect(mapped, originalSize.Width, originalSize.Height);
                    all.Add(new DetectionResult
                    {
                        BatchId = b,
                        ClassId = det.ClassId,
                        ClassDescribe = (classNames != null && det.ClassId < classNames.Length) ? classNames[det.ClassId] : $"Class_{det.ClassId}",
                        Confidence = det.Confidence,
                        BoundingBox = mapped
                    });
                }
            }
            // 全局 NMS
            return ApplyClassAwareNMS(all, classNames, iouThreshold);
        }
        #endregion

        #region Inference（遗留 Obsolete）
        /// <summary>
        /// 执行一次推理（旧版 Obsolete）。
        /// </summary>
        /// <param name="inputtensor">输入张量描述。</param>
        /// <param name="outputtensor">输出张量描述。</param>
        /// <param name="mat">单张输入图像。</param>
        /// <returns>封装 GPU 推理结果的 <c>InferenceTensor</c>。</returns>
        /// <exception cref="Exception">当底层推理器未初始化时。</exception>
        [Obsolete("请改用 Detect（同步）或 DetectAsync（异步）2025.9.2弃用")]
        public unsafe InferenceTensor InFerencePtr(ref InputTensor inputtensor, ref OutputTensor outputtensor, Mat mat)
        {
            if (NVIDIAPredictor == null)
                throw new Exception("NVIDIAPredictor初始化失败！");
            List<Mat> values = new List<Mat> { mat };
            Stopwatch totalWatch = Stopwatch.StartNew();
            Stopwatch sw = Stopwatch.StartNew();
            var intptr = ProcessBatch(ref inputtensor, values);
            sw.Stop();
            long processBatchTime = sw.ElapsedMilliseconds;
            sw.Restart();
            NVIDIAPredictor.LoadInferenceData("images", intptr.DataPtr, intptr.Length);
            sw.Stop();
            long loadDataTime = sw.ElapsedMilliseconds;
            sw.Restart();
            NVIDIAPredictor.Infer();
            sw.Stop();
            long inferTime = sw.ElapsedMilliseconds;
            sw.Restart();
            InferenceTensor outputData = NVIDIAPredictor.GetInferenceResultPtr(ref outputtensor, "output0");
            intptr.Dispose();
            sw.Stop();
            long getResultTime = sw.ElapsedMilliseconds;
            totalWatch.Stop();
            long totalTime = totalWatch.ElapsedMilliseconds;
            DebugLog($"[ProcessBatch-2] 性能统计 - 预处理: {processBatchTime}ms, 加载: {loadDataTime}ms, 推理: {inferTime}ms, 结果: {getResultTime}ms, 总计: {totalTime}ms");
            return outputData;
        }

        /// <summary>
        /// 执行批量推理（GPU 内存访问，旧版 Obsolete）。
        /// </summary>
        /// <param name="inputtensor">输入张量描述。</param>
        /// <param name="outputtensor">输出张量描述。</param>
        /// <param name="values">批量输入图像。</param>
        /// <returns>零拷贝输出指针。</returns>
        /// <exception cref="Exception">当底层推理器未初始化时。</exception>
        [Obsolete("请改用 Detect（同步）或 DetectAsync（异步）2025.9.2弃用")]
        public IntPtr InFerencePtr(ref InputTensor inputtensor, ref OutputTensor outputtensor, List<Mat> values)
        {
            if (NVIDIAPredictor == null)
                throw new Exception("NVIDIAPredictor初始化失败！");
            Stopwatch totalWatch = Stopwatch.StartNew();
            Stopwatch sw = Stopwatch.StartNew();
            var intptr = ProcessBatch(ref inputtensor, values);
            sw.Stop();
            long processBatchTime = sw.ElapsedMilliseconds;
            sw.Restart();
            NVIDIAPredictor.LoadInferenceData("images", intptr.DataPtr, intptr.Length);
            sw.Stop();
            long loadDataTime = sw.ElapsedMilliseconds;
            sw.Restart();
            NVIDIAPredictor.Infer();
            sw.Stop();
            long inferTime = sw.ElapsedMilliseconds;
            sw.Restart();
            var intPtr = NVIDIAPredictor.GetZeroCopyResult(ref outputtensor, "output0");
            intptr.Dispose();
            sw.Stop();
            long getResultTime = sw.ElapsedMilliseconds;
            totalWatch.Stop();
            long totalTime = totalWatch.ElapsedMilliseconds;
            DebugLog($"[ProcessBatch-3] 性能统计 - 预处理: {processBatchTime}ms, 加载: {loadDataTime}ms, 推理: {inferTime}ms, 结果: {getResultTime}ms, 总计: {totalTime}ms");
            return intPtr;
        }
        #endregion

        #region Parsing（非 SAHI）
        /// <summary>
        /// 计算输出张量展平后的线性索引（按 [batch, channel, detection]）。
        /// </summary>
        /// <param name="batch">批量索引。</param>
        /// <param name="channel">通道索引。</param>
        /// <param name="detection">检测项索引。</param>
        /// <param name="shape">输出形状。</param>
        /// <returns>线性索引。</returns>
        private int GetIndex(int batch, int channel, int detection, OutputTensor shape)
        {
            return batch * shape.Channels * shape.NumDetections
                 + channel * shape.NumDetections
                 + detection;
        }

        /// <summary>
        /// 解析模型的输出张量（指针版）。
        /// </summary>
        /// <param name="output">输出数据指针。</param>
        /// <param name="inshape">输入张量描述。</param>
        /// <param name="outshape">输出张量描述。</param>
        /// <param name="yamlpath">YOLO yaml 路径。</param>
        /// <param name="confidenceThreshold">置信度阈值。</param>
        /// <param name="iouThreshold">NMS IoU 阈值。</param>
        /// <returns>单图的检测结果（按类 NMS）。</returns>
        public unsafe List<DetectionResult> ParseYoloOutput(
            float* output,
            InputTensor inshape,
            OutputTensor outshape,
            string yamlpath,
            float confidenceThreshold = 0.3f,
            float iouThreshold = 0.3f)
        {
            if (output == null) throw new ArgumentNullException(nameof(output), "输出数据不能为空！");
            int channels = outshape.Channels;
            int numDetections = outshape.NumDetections;
            if (channels < 5) throw new ArgumentException($"通道大小错误：{channels}！");
            int numClasses = channels - 4; // 固定布局：x,y,w,h,cls...
            DebugLog($"[ParseYoloOutput] 开始解析YOLO输出(指针) - 通道数: {channels}, 检测数: {numDetections}, 类别数: {numClasses}");
            var results = new List<DetectionResult>(numDetections);
            (float scale, float padX, float padY) = CachedCalculatePreprocessParams(inshape, inshape.InputImageSize[0]);
            DebugLog($"[ParseYoloOutput] 坐标还原参数 - 缩放: {scale:F5}, 填充: ({padX:F2},{padY:F2}), 原图: {inshape.InputImageSize[0].Width}x{inshape.InputImageSize[0].Height}");
            for (int i = 0; i < numDetections; i++)
            {
                float cx = *(output + GetIndex(0, 0, i, outshape));
                float cy = *(output + GetIndex(0, 1, i, outshape));
                float w = *(output + GetIndex(0, 2, i, outshape));
                float h = *(output + GetIndex(0, 3, i, outshape));
                int bestClassId = -1;
                float maxClassScore = float.MinValue;
                for (int c = 0; c < numClasses; c++)
                {
                    float score = *(output + GetIndex(0, 4 + c, i, outshape));
                    if (score > maxClassScore)
                    {
                        maxClassScore = score;
                        bestClassId = c;
                    }
                }
                if (DebugParsing && i < DebugMaxDetPrints)
                {
                    DebugLog($"[ParseYoloOutput] 检测{i}: 原始坐标=({cx:F2},{cy:F2},{w:F2},{h:F2}), 类别={bestClassId}, 置信度={maxClassScore:F4}");
                }
                if (maxClassScore < confidenceThreshold)
                    continue;
                RectF rect = ConvertToOriginalCoords(cx, cy, w, h, scale, padX, padY, inshape.InputImageSize[0]);
                rect = ClampRect(rect, inshape.InputImageSize[0].Width, inshape.InputImageSize[0].Height);
                if (DebugParsing && i < DebugMaxDetPrints)
                {
                    DebugLog($"[ParseYoloOutput] 检测{i}: 还原后坐标=({rect.X:F1},{rect.Y:F1},{rect.Width:F1},{rect.Height:F1})");
                }
                results.Add(new DetectionResult
                {
                    ClassId = bestClassId,
                    Confidence = maxClassScore,
                    BoundingBox = rect
                });
            }
            var classdescribe = GetClassNamesFromYaml(yamlpath);
            var before = results.Count;
            var nmsed = ApplyClassAwareNMS(results, classdescribe, iouThreshold);
            DebugLog($"[ParseYoloOutput] NMS完成 - 输入: {before}, 输出: {nmsed.Count}, IoU阈值: {iouThreshold}");
            return nmsed;
        }

        /// <summary>
        /// 解析模型的输出张量（托管数组版，Obsolete）。
        /// </summary>
        /// <param name="output">输出数组。</param>
        /// <param name="inshape">输入张量描述。</param>
        /// <param name="outshape">输出张量描述。</param>
        /// <param name="originalSize">原图尺寸。</param>
        /// <param name="yamlpath">YOLO yaml 路径。</param>
        /// <param name="confidenceThreshold">置信度阈值。</param>
        /// <param name="iouThreshold">NMS IoU 阈值。</param>
        /// <returns>单图检测结果（按类 NMS）。</returns>
        [Obsolete("请改用 Detect / DetectAsync 统一入口；如需单独解析请使用指针版本或 SAHI 全局解析方法 2025.9.2弃用")]
        public List<DetectionResult> ParseYoloOutput(
            float[] output,
            InputTensor inshape,
            OutputTensor outshape,
            OpenCvSharp.Size originalSize,
            string yamlpath,
            float confidenceThreshold = 0.3f,
            float iouThreshold = 0.3f)
        {
            if (output == null) throw new ArgumentNullException(nameof(output), "输出数据不能为空！");
            int channels = outshape.Channels;
            int numDetections = outshape.NumDetections;
            if (channels < 5) throw new ArgumentException($"通道大小错误：{channels}！");
            int numClasses = channels - 4; // 固定布局：x,y,w,h,cls...
            DebugLog($"[ParseYoloOutput] 开始解析YOLO输出(数组) - 通道数: {channels}, 检测数: {numDetections}, 类别数: {numClasses}");
            var results = new List<DetectionResult>(numDetections);
            (float scale, float padX, float padY) = CachedCalculatePreprocessParams(inshape, originalSize);
            DebugLog($"[ParseYoloOutput] 坐标还原参数 - 缩放: {scale:F5}, 填充: ({padX:F2},{padY:F2}), 原图: {originalSize.Width}x{originalSize.Height}");
            for (int i = 0; i < numDetections; i++)
            {
                float cx = output[GetIndex(0, 0, i, outshape)];
                float cy = output[GetIndex(0, 1, i, outshape)];
                float w = output[GetIndex(0, 2, i, outshape)];
                float h = output[GetIndex(0, 3, i, outshape)];
                int bestClassId = -1;
                float maxClassScore = float.MinValue;
                for (int c = 0; c < numClasses; c++)
                {
                    float score = output[GetIndex(0, 4 + c, i, outshape)];
                    if (score > maxClassScore)
                    {
                        maxClassScore = score;
                        bestClassId = c;
                    }
                }
                if (DebugParsing && i < DebugMaxDetPrints)
                {
                    DebugLog($"[ParseYoloOutput] 检测{i}: 原始坐标=({cx:F2},{cy:F2},{w:F2},{h:F2}), 类别={bestClassId}, 置信度={maxClassScore:F4}");
                }
                if (maxClassScore < confidenceThreshold)
                    continue;
                RectF rect = ConvertToOriginalCoords(cx, cy, w, h, scale, padX, padY, originalSize);
                rect = ClampRect(rect, originalSize.Width, originalSize.Height);
                if (DebugParsing && i < DebugMaxDetPrints)
                {
                    DebugLog($"[ParseYoloOutput] 检测{i}: 还原后坐标=({rect.X:F1},{rect.Y:F1},{rect.Width:F1},{rect.Height:F1})");
                }
                results.Add(new DetectionResult
                {
                    ClassId = bestClassId,
                    Confidence = maxClassScore,
                    BoundingBox = rect
                });
            }
            var classdescribe = GetClassNamesFromYaml(yamlpath);
            var before = results.Count;
            var nmsed = ApplyClassAwareNMS(results, classdescribe, iouThreshold);
            DebugLog($"[ParseYoloOutput] NMS完成 - 输入: {before}, 输出: {nmsed.Count}, IoU阈值: {iouThreshold}");
            return nmsed;
        }

        /// <summary>
        /// 解析 YOLO 模型的输出张量（批量，Obsolete）。
        /// </summary>
        /// <param name="output">输出指针。</param>
        /// <param name="inshape">输入张量描述。</param>
        /// <param name="outshape">输出张量描述。</param>
        /// <param name="yamlpath">YOLO yaml 路径。</param>
        /// <param name="confidenceThreshold">置信度阈值。</param>
        /// <param name="iouThreshold">NMS IoU 阈值。</param>
        /// <returns>按 batch 的检测结果集合（各自已做 NMS）。</returns>
        [Obsolete("请优先使用 Detect / DetectAsync；批量解析可封装为多次调用或自定义回调处理 2025.9.2弃用")]
        public unsafe List<List<DetectionResult>> BatchParseYoloOutput(
                float* output,
                InputTensor inshape,
                OutputTensor outshape,
                string yamlpath,
                float confidenceThreshold = 0.3f,
                float iouThreshold = 0.3f)
        {
            if (output == null) throw new ArgumentNullException(nameof(output), "输出数据不能为空！");
            int batchSize = outshape.BatchSize;
            int channels = outshape.Channels;
            int numDetections = outshape.NumDetections;
            if (channels < 5) throw new ArgumentException($"通道大小错误：{channels}！");
            int numClasses = channels - 4; // 固定布局：x,y,w,h,cls...
            DebugLog($"[BatchParseYoloOutput] 开始批量解析YOLO输出 - 批次: {batchSize}, 通道数: {channels}, 检测数: {numDetections}, 类别数: {numClasses}");
            var allResults = new Dictionary<int, List<DetectionResult>>();
            Parallel.For(0, batchSize, b =>
            {
                (float scale, float padX, float padY) = CachedCalculatePreprocessParams(inshape, inshape.InputImageSize[b]);
                var results = new List<DetectionResult>(numDetections);
                for (int i = 0; i < numDetections; i++)
                {
                    float cx = *(output + GetIndex(b, 0, i, outshape));
                    float cy = *(output + GetIndex(b, 1, i, outshape));
                    float w = *(output + GetIndex(b, 2, i, outshape));
                    float h = *(output + GetIndex(b, 3, i, outshape));
                    int bestClassId = -1;
                    float maxClassScore = float.MinValue;
                    for (int c = 0; c < numClasses; c++)
                    {
                        float score = *(output + GetIndex(b, 4 + c, i, outshape));
                        if (score > maxClassScore)
                        {
                            maxClassScore = score;
                            bestClassId = c;
                        }
                    }
                    if (maxClassScore < confidenceThreshold)
                        continue;
                    RectF rect = ConvertToOriginalCoords(cx, cy, w, h, scale, padX, padY, inshape.InputImageSize[b]);
                    rect = ClampRect(rect, inshape.InputImageSize[b].Width, inshape.InputImageSize[b].Height);
                    lock (allResults)
                    {
                        results.Add(new DetectionResult
                        {
                            BatchId = b,
                            ClassId = bestClassId,
                            Confidence = maxClassScore,
                            BoundingBox = rect
                        });
                    }
                }
                lock (allResults)
                {
                    allResults[b] = results;
                }
            });
            var classdescribe = GetClassNamesFromYaml(yamlpath);
            var finalResults = new List<List<DetectionResult>>();
            var sortedResults = allResults.OrderBy(kvp => kvp.Key).ToList();
            foreach (var item in sortedResults)
            {
                var result = ApplyClassAwareNMS(item.Value, classdescribe, iouThreshold);
                finalResults.Add(result);
                if (DebugParsing)
                {
                    DebugLog($"[BatchParseYoloOutput] 批次{item.Key}NMS结果: 输入={item.Value.Count}, 输出={result.Count}");
                }
            }
            return finalResults;
        }
        #endregion

        #region Geometry & NMS Utilities
        /// <summary>
        /// 计算预处理过程中的缩放和填充参数（等比缩放 + 居中填充）。
        /// </summary>
        /// <param name="input">输入张量描述（目标宽高）。</param>
        /// <param name="originalSize">原图尺寸。</param>
        /// <returns>返回缩放因子与 X/Y 方向填充像素。</returns>
        private static (float scale, float padX, float padY) CalculatePreprocessParams(InputTensor input, OpenCvSharp.Size originalSize)
        {
            float width = originalSize.Width;
            float height = originalSize.Height;
            float scaleW = (float)input.Width / width;
            float scaleH = (float)input.Height / height;
            float scale = Math.Min(scaleW, scaleH);
            float scaledWidth = width * scale;
            float scaledHeight = height * scale;
            float padX = (input.Width - scaledWidth) * 0.5f;
            float padY = (input.Height - scaledHeight) * 0.5f;
            return (scale, padX, padY);
        }

        /// <summary>
        /// 将模型输出的坐标（输入尺寸坐标系）转换为原始图像坐标系。
        /// </summary>
        /// <param name="cx">中心 X（输入尺寸坐标系）。</param>
        /// <param name="cy">中心 Y。</param>
        /// <param name="w">宽。</param>
        /// <param name="h">高。</param>
        /// <param name="scale">等比缩放因子。</param>
        /// <param name="padX">X 方向填充。</param>
        /// <param name="padY">Y 方向填充。</param>
        /// <param name="originalSize">原图尺寸。</param>
        /// <returns>原图坐标系下的矩形。</returns>
        private static RectF ConvertToOriginalCoords(
            float cx, float cy, float w, float h,
            float scale, float padX, float padY,
            OpenCvSharp.Size originalSize)
        {
            float cx_unpadded = cx - padX;
            float cy_unpadded = cy - padY;
            float w_unpadded = w;
            float h_unpadded = h;
            float invScale = 1.0f / scale;
            float cx_original = cx_unpadded * invScale;
            float cy_original = cy_unpadded * invScale;
            float w_original = w_unpadded * invScale;
            float h_original = h_unpadded * invScale;
            float halfW = 0.5f * w_original;
            float halfH = 0.5f * h_original;
            float x1 = cx_original - halfW;
            float y1 = cy_original - halfH;
            float x2 = cx_original + halfW;
            float y2 = cy_original + halfH;
            float maxW = originalSize.Width;
            float maxH = originalSize.Height;
            x1 = Clamp(x1, 0, maxW);
            y1 = Clamp(y1, 0, maxH);
            x2 = Clamp(x2, 0, maxW);
            y2 = Clamp(y2, 0, maxH);
            return new RectF(x1, y1, x2 - x1, y2 - y1);
        }

        /// <summary>
        /// 夹取数值到 [min, max] 范围。
        /// </summary>
        /// <param name="val">输入值。</param>
        /// <param name="min">最小值。</param>
        /// <param name="max">最大值。</param>
        /// <returns>夹取后的值。</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Clamp(float val, float min, float max)
        {
            return (val < min) ? min : (val > max) ? max : val;
        }

        /// <summary>
        /// 将矩形裁剪到给定尺寸内。
        /// </summary>
        /// <param name="rect">输入矩形。</param>
        /// <param name="maxWidth">最大宽度。</param>
        /// <param name="maxHeight">最大高度。</param>
        /// <returns>裁剪后的矩形。</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private RectF ClampRect(RectF rect, int maxWidth, int maxHeight)
        {
            float x = rect.X;
            float y = rect.Y;
            float width = rect.Width;
            float height = rect.Height;
            x = x < 0f ? 0f : (x > maxWidth ? maxWidth : x);
            y = y < 0f ? 0f : (y > maxHeight ? maxHeight : y);
            float maxWidthForRect = maxWidth - x;
            float maxHeightForRect = maxHeight - y;
            width = width < 0f ? 0f : (width > maxWidthForRect ? maxWidthForRect : width);
            height = height < 0f ? 0f : (height > maxHeightForRect ? maxHeightForRect : height);
            return new RectF(x, y, width, height);
        }

        /// <summary>
        /// 计算两个矩形的 IoU（交并比）。
        /// </summary>
        /// <param name="a">矩形 A。</param>
        /// <param name="b">矩形 B。</param>
        /// <returns>IoU 值，范围 [0,1]。</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float CalculateIoU(RectF a, RectF b)
        {
            float interX1 = a.X > b.X ? a.X : b.X;
            float interY1 = a.Y > b.Y ? a.Y : b.Y;
            float interX2 = (a.X + a.Width) < (b.X + b.Width) ? (a.X + a.Width) : (b.X + b.Width);
            float interY2 = (a.Y + a.Height) < (b.Y + b.Height) ? (a.Y + a.Height) : (b.Y + b.Height);
            float interWidth = interX2 - interX1;
            float interHeight = interY2 - interY1;
            if (interWidth <= 0f || interHeight <= 0f)
                return 0f;
            float interArea = interWidth * interHeight;
            float unionArea = a.Width * a.Height + b.Width * b.Height - interArea;
            return unionArea > 0f ? interArea / unionArea : 0f;
        }

        /// <summary>
        /// NMS（按类别分组）。
        /// </summary>
        /// <param name="detections">待筛选的检测结果。</param>
        /// <param name="classdescribe">类别名数组，可为 null。</param>
        /// <param name="iouThreshold">IoU 阈值。</param>
        /// <returns>经过按类 NMS 的检测结果集合。</returns>
        private List<DetectionResult> ApplyClassAwareNMS(
            List<DetectionResult> detections,
            string[] classdescribe,
            float iouThreshold)
        {
            if (detections == null || detections.Count == 0)
            {
                if (DebugParsing) VerboseLog("[ApplyClassAwareNMS] 输入检测结果为空");
                return new List<DetectionResult>(0);
            }

            DebugLog($"[ApplyClassAwareNMS] 开始NMS - 输入检测数: {detections.Count}, IoU阈值: {iouThreshold}");

            var finalResults = new List<DetectionResult>(detections.Count);
            var classGroups = new Dictionary<int, List<DetectionResult>>();
            foreach (var det in detections)
            {
                if (!classGroups.TryGetValue(det.ClassId, out var list))
                {
                    list = new List<DetectionResult>();
                    classGroups[det.ClassId] = list;
                }
                list.Add(det);
            }

            VerboseLog($"[ApplyClassAwareNMS] 分类统计 - 类别数: {classGroups.Count}");
            if (DebugParsing)
            {
                var pre = detections.Count;
                VerboseLog($"[ApplyClassAwareNMS] NMS预处理 - 输入检测数: {pre}, 类别组数: {classGroups.Count}, IoU阈值: {iouThreshold}");
            }
            foreach (var kvp in classGroups)
            {
                var classId = kvp.Key;
                var classDetections = kvp.Value;
                classDetections.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
                int count = classDetections.Count;
                if (DebugParsing && count > 0)
                    VerboseLog($"[ApplyClassAwareNMS] 类别{classId}: {count}个检测，最高置信度: {classDetections[0].Confidence:F3}");

                bool[] suppressed = new bool[count];
                int suppressedCount = 0;
                for (int i = 0; i < count; i++)
                {
                    if (suppressed[i]) continue;
                    var current = classDetections[i];
                    if (classdescribe != null && current.ClassId < classdescribe.Length)
                        current.ClassDescribe = classdescribe[current.ClassId];
                    else
                        current.ClassDescribe = $"Class_{current.ClassId}"; // 提供默认类别名
                    finalResults.Add(current);
                    RectF boxA = current.BoundingBox;
                    for (int j = i + 1; j < count; j++)
                    {
                        if (suppressed[j]) continue;
                        float iou = CalculateIoU(boxA, classDetections[j].BoundingBox);
                        if (iou > iouThreshold)
                        {
                            suppressed[j] = true;
                            suppressedCount++;
                        }
                    }
                }

                if (DebugParsing && suppressedCount > 0)
                    VerboseLog($"[ApplyClassAwareNMS] 类别{classId}: 抑制了{suppressedCount}个重复检测");
            }

            DebugLog($"[ApplyClassAwareNMS] NMS完成 - 输入: {detections.Count}, 输出: {finalResults.Count}");
            return finalResults;
        }

        /// <summary>
        /// 解析 YAML 文件获取类名信息。
        /// </summary>
        /// <param name="yamlFilePath">YOLO 数据集 yaml 文件路径。</param>
        /// <returns>类别名数组；当路径无效时返回 null。</returns>
        /// <exception cref="RegexMatchTimeoutException">当 yaml 内容未匹配到 names 字段时。</exception>
        public string[] GetClassNamesFromYaml(string yamlFilePath)
        {
            if (YamlClass == null)
            {
                DebugLog($"[GetClassNamesFromYaml] 开始解析YAML文件: {yamlFilePath}");
                if (yamlFilePath == null || !File.Exists(yamlFilePath))
                {
                    DebugLog("[GetClassNamesFromYaml] YAML文件不存在，返回null");
                    return null;
                }
                var yamlContent = File.ReadAllText(yamlFilePath);
                DebugLog($"[GetClassNamesFromYaml] YAML文件长度: {yamlContent.Length}");
                var match = Regex.Match(yamlContent, @"names:\s*\[([^\]]+)\]");
                if (match.Success)
                {
                    var namesString = match.Groups[1].Value;
                    YamlClass = namesString.Split(',')
                                                  .Select(name => name.Trim().Trim('\'', '"'))
                                                  .ToArray();
                    DebugLog($"[GetClassNamesFromYaml] 成功解析类别名称，数量: {YamlClass.Length}");
                    _yamlClassPath = yamlFilePath;
                    return YamlClass;
                }
                else
                {
                    DebugLog("[GetClassNamesFromYaml] YAML解析失败，未找到names字段");
                    throw new RegexMatchTimeoutException(match.Name);
                }
            }
            else
            {
                DebugLog($"[GetClassNamesFromYaml] 使用缓存的类别名称，数量: {YamlClass.Length}");
                return YamlClass;
            }
        }
        #endregion

        #region Preprocess & Packing
        /// <summary>
        /// 预处理模型输入张量数据：将多张 BGR Mat 统一等比缩放+填充，并打包为 NCHW float 数组。
        /// </summary>
        /// <param name="inputtensor">输入张量描述（目标宽高、通道）。</param>
        /// <param name="images">输入图像列表。</param>
        /// <returns>可直接传入推理端的 <c>SafeTensor</c>（需 <c>Dispose()</c> 释放）。</returns>
        /// <exception cref="Exception">当输入列表为空或图像无效时。</exception>
        private unsafe SafeTensor ProcessBatch(ref InputTensor inputtensor, List<Mat> images)
        {
            if (images == null || images.Count < 1)
                throw new Exception("输入images错误");

            int actualBatch = images.Count;
            int singleImageSize = inputtensor.Channels * inputtensor.Height * inputtensor.Width;
            DebugLog($"[ProcessBatch] 开始批量预处理 - 批次: {actualBatch}, 单图大小: {singleImageSize}, 总大小: {actualBatch * singleImageSize}");

            float[] batchData = GetBuffer(actualBatch * singleImageSize);
            try
            {
                OpenCvSharp.Size[] InputImageSize = new OpenCvSharp.Size[actualBatch];
                int Width = inputtensor.Width;
                int Height = inputtensor.Height;
                int Channels = inputtensor.Channels;
                DebugLog($"[ProcessBatch] 目标尺寸: {Width}x{Height}x{Channels}");

                var handle = GCHandle.Alloc(batchData, GCHandleType.Pinned);
                try
                {
                    IntPtr basePtr = handle.AddrOfPinnedObject();
                    DebugLog("[ProcessBatch] 开始并行预处理图像");
                    Parallel.For(0, actualBatch, i =>
                    {
                        InputImageSize[i] = images[i].Size();
                        if (DebugParsing && i < DebugMaxDetPrints)
                            DebugLog($"[ProcessBatch] 处理图像 {i}: {images[i].Width}x{images[i].Height} -> {Width}x{Height}");
                        using (var processed = OptimizedPreprocessImage(images[i], Width, Height))
                        {
                            int offset = i * singleImageSize;
                            if (processed.Total() != singleImageSize)
                                throw new InvalidOperationException($"图像大小不匹配 {singleImageSize}, got {processed.Total()}");
                            fixed (float* pDest = &batchData[offset])
                            {
                                Buffer.MemoryCopy(processed.DataPointer, pDest, processed.Total() * sizeof(float), processed.Total() * sizeof(float));
                            }
                        }
                    });
                }
                finally
                {
                    if (handle.IsAllocated) handle.Free();
                }

                inputtensor.InputImageSize = InputImageSize;
                var tensor = new SafeTensor(batchData, new[] { actualBatch, inputtensor.Channels, Height, Width });
                return tensor;
            }
            catch (Exception)
            {
                ReturnBuffer(batchData);
                throw;
            }
        }

        /// <summary>
        /// 优化的图像预处理方法，使用Mat池和内联内存复制
        /// </summary>
        /// <param name="image">输入 BGR Mat</param>
        /// <param name="targetWidth">目标宽度</param>
        /// <param name="targetHeight">目标高度</param>
        /// <returns>预处理后的 DNN blob</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Mat OptimizedPreprocessImage(Mat image, int targetWidth, int targetHeight)
        {
            if (image == null || image.Empty())
                throw new ArgumentException("输入图像不能为空");

            int originalWidth = image.Width;
            int originalHeight = image.Height;

            if (DebugParsing && originalWidth != targetWidth && originalHeight != targetHeight)
                DebugLog($"[OptimizedPreprocess] 图像缩放 {originalWidth}x{originalHeight} -> {targetWidth}x{targetHeight}");

            // 使用缓存的预处理参数
            float scaleW = (float)targetWidth / originalWidth;
            float scaleH = (float)targetHeight / originalHeight;
            float scale = Math.Min(scaleW, scaleH); float scaledWidth = originalWidth * scale;
            float scaledHeight = originalHeight * scale;
            float padX = (targetWidth - scaledWidth) * 0.5f;
            float padY = (targetHeight - scaledHeight) * 0.5f;

            int scaledWidthInt = (int)Math.Round(scaledWidth);
            int scaledHeightInt = (int)Math.Round(scaledHeight);
            int padXInt = (int)Math.Round(padX);
            int padYInt = (int)Math.Round(padY);

            // 使用Mat池优化内存分配
            Mat resized = GetMat(scaledHeightInt, scaledWidthInt, MatType.CV_8UC3);
            Mat padded = GetMat(targetHeight, targetWidth, MatType.CV_8UC3, new Scalar(114, 114, 114));

            try
            {
                Cv2.Resize(image, resized, new OpenCvSharp.Size(scaledWidthInt, scaledHeightInt));

                Rect roi = new Rect(padXInt, padYInt, scaledWidthInt, scaledHeightInt);
                using (var paddedRoi = new Mat(padded, roi))
                {
                    resized.CopyTo(paddedRoi);
                }

                Mat result = CvDnn.BlobFromImage(
                    padded,
                    1.0 / 255.0,
                    new OpenCvSharp.Size(targetWidth, targetHeight),
                    new Scalar(0, 0, 0),
                    swapRB: true,
                    crop: false);

                return result;
            }
            finally
            {
                ReturnMat(resized);
                ReturnMat(padded);
            }
        }

        /// <summary>
        /// 预处理单张图像，保持长宽比并进行居中填充（返回用于 DNN 的 blob）。
        /// </summary>
        /// <param name="image">输入 BGR Mat。</param>
        /// <param name="targetWidth">目标宽。</param>
        /// <param name="targetHeight">目标高。</param>
        /// <returns>OpenCV DNN blob（CHW float）。</returns>
        /// <exception cref="ArgumentException">当输入图像为空时。</exception>
        private static Mat PreprocessImageKeepAspectRatio(Mat image, int targetWidth, int targetHeight)
        {
            if (image == null || image.Empty())
                throw new ArgumentException("输入图像不能为空");
            int originalWidth = image.Width;
            int originalHeight = image.Height;
            float scaleW = (float)targetWidth / originalWidth;
            float scaleH = (float)targetHeight / originalHeight;
            float scale = Math.Min(scaleW, scaleH);
            float scaledWidth = originalWidth * scale;
            float scaledHeight = originalHeight * scale;
            float padX = (targetWidth - scaledWidth) * 0.5f;
            float padY = (targetHeight - scaledHeight) * 0.5f;
            int scaledWidthInt = (int)Math.Round(scaledWidth);
            int scaledHeightInt = (int)Math.Round(scaledHeight);
            int padXInt = (int)Math.Round(padX);
            int padYInt = (int)Math.Round(padY);
            Mat resized = new Mat();
            Cv2.Resize(image, resized, new OpenCvSharp.Size(scaledWidthInt, scaledHeightInt));
            Mat padded = new Mat(targetHeight, targetWidth, MatType.CV_8UC3, new Scalar(114, 114, 114));
            Rect roi = new Rect(padXInt, padYInt, scaledWidthInt, scaledHeightInt);
            resized.CopyTo(new Mat(padded, roi));
            resized.Dispose();
            Mat result = CvDnn.BlobFromImage(
                padded,
                1.0 / 255.0,
                new OpenCvSharp.Size(targetWidth, targetHeight),
                new Scalar(0, 0, 0),
                swapRB: true,
                crop: false);

            padded.Dispose();
            return result;
        }

        /// <summary>
        /// 等比缩放并填充到目标尺寸，返回 CV_8UC3 图像（常用于 SAHI tile 打包前的准备）。
        /// </summary>
        /// <param name="image">输入 BGR Mat。</param>
        /// <param name="targetWidth">目标宽。</param>
        /// <param name="targetHeight">目标高。</param>
        /// <returns>经等比缩放并居中填充的 CV_8UC3 Mat。</returns>
        /// <exception cref="ArgumentException">当输入图像为空时。</exception>
        private static Mat PreprocessImageKeepAspectRatioPadded(Mat image, int targetWidth, int targetHeight)
        {
            if (image == null || image.Empty())
                throw new ArgumentException("输入图像不能为空");

            int originalWidth = image.Width;
            int originalHeight = image.Height;

            float scaleW = (float)targetWidth / originalWidth;
            float scaleH = (float)targetHeight / originalHeight;
            float scale = Math.Min(scaleW, scaleH);

            // 使用一次仿射变换完成缩放+平移（填充）
            float tx = (targetWidth - originalWidth * scale) * 0.5f;
            float ty = (targetHeight - originalHeight * scale) * 0.5f;
            Mat M = new Mat(2, 3, MatType.CV_64FC1);
            // [ s, 0, tx; 0, s, ty ]
            M.Set(0, 0, (double)scale); M.Set(0, 1, 0.0); M.Set(0, 2, (double)tx);
            M.Set(1, 0, 0.0); M.Set(1, 1, (double)scale); M.Set(1, 2, (double)ty);

            Mat padded = new Mat();
            Cv2.WarpAffine(
                image,
                padded,
                M,
                new OpenCvSharp.Size(targetWidth, targetHeight),
                InterpolationFlags.Linear,
                BorderTypes.Constant,
                new Scalar(114, 114, 114)
            );
            M.Dispose();
            return padded;
        }

        /// <summary>
        /// 将若干张 CV_8UC3 tiles 打包为 float NCHW（批量×通道×高×宽）。
        /// </summary>
        /// <param name="tiles">CV_8UC3 tiles 数组（尺寸应与目标一致）。</param>
        /// <param name="targetWidth">目标宽。</param>
        /// <param name="targetHeight">目标高。</param>
        /// <param name="dest">目标 float 数组（长度需满足 B×C×H×W）。</param>
        /// <param name="channels">通道数（通常为 3）。</param>
        /// <param name="swapRB">是否交换 R/B 通道。</param>
        /// <param name="scale">缩放系数（如 1/255）。</param>
        private static unsafe void PackTilesToNCHWFloat(Mat[] tiles, int targetWidth, int targetHeight, float[] dest,
            int channels, bool swapRB, float scale)
        {
            if (channels != 3)
                throw new NotSupportedException($"仅支持 3 通道，当前: {channels}");

            int batch = tiles.Length;
            int singleSize = channels * targetHeight * targetWidth;

            var handle = GCHandle.Alloc(dest, GCHandleType.Pinned);
            try
            {
                IntPtr basePtr = handle.AddrOfPinnedObject();
                Parallel.For(0, batch, b =>
                {
                    var tile = tiles[b];
                    if (tile.Empty()) throw new ArgumentException($"第 {b} 张图为空");
                    if (tile.Type() != MatType.CV_8UC3)
                        throw new NotSupportedException($"第 {b} 张图类型须为 CV_8UC3，当前: {tile.Type()}");
                    if (tile.Width != targetWidth || tile.Height != targetHeight)
                        throw new ArgumentException($"第 {b} 张图尺寸不匹配，期望 {targetWidth}x{targetHeight}，实际 {tile.Width}x{tile.Height}");

                    byte* src = tile.DataPointer;
                    int srcStride = (int)tile.Step(); // bytes per row

                    float* pBase = (float*)basePtr.ToPointer();
                    int planeSize = targetHeight * targetWidth;
                    float* dstB = pBase + b * singleSize + 0 * planeSize;
                    float* dstG = pBase + b * singleSize + 1 * planeSize;
                    float* dstR = pBase + b * singleSize + 2 * planeSize;

                    // 每像素 3 通道，OpenCV 默认 BGR 排列
                    if (swapRB)
                    {
                        for (int y = 0; y < targetHeight; y++)
                        {
                            byte* row = src + y * srcStride;
                            int rowOffset = y * targetWidth;
                            float* dR = dstR + rowOffset;
                            float* dG = dstG + rowOffset;
                            float* dB = dstB + rowOffset;
                            for (int x = 0; x < targetWidth; x++)
                            {
                                int pxOffset = x * 3;
                                byte bgrB = row[pxOffset + 0];
                                byte bgrG = row[pxOffset + 1];
                                byte bgrR = row[pxOffset + 2];
                                *dR++ = bgrR * scale;
                                *dG++ = bgrG * scale;
                                *dB++ = bgrB * scale;
                            }
                        }
                    }
                    else
                    {
                        for (int y = 0; y < targetHeight; y++)
                        {
                            byte* row = src + y * srcStride;
                            int rowOffset = y * targetWidth;
                            float* dB = dstB + rowOffset;
                            float* dG = dstG + rowOffset;
                            float* dR = dstR + rowOffset;
                            for (int x = 0; x < targetWidth; x++)
                            {
                                int pxOffset = x * 3;
                                byte bgrB = row[pxOffset + 0];
                                byte bgrG = row[pxOffset + 1];
                                byte bgrR = row[pxOffset + 2];
                                *dB++ = bgrB * scale;
                                *dG++ = bgrG * scale;
                                *dR++ = bgrR * scale;
                            }
                        }
                    }
                });
            }
            finally
            {
                if (handle.IsAllocated) handle.Free();
            }
        }

        /// <summary>
        /// 将单张经等比缩放与填充后的 CV_8UC3 瓦片拷贝到目标 NCHW float 缓冲区的指定偏移。
        /// </summary>
        /// <param name="basePtr">目标缓冲区基指针（float*）。</param>
        /// <param name="destOffsetFloats">写入偏移（以 float 为单位）。</param>
        /// <param name="padded">输入 CV_8UC3 Mat（尺寸为目标大小）。</param>
        /// <param name="targetWidth">目标宽。</param>
        /// <param name="targetHeight">目标高。</param>
        /// <param name="channels">通道数（3）。</param>
        /// <param name="swapRB">是否交换 R/B 通道。</param>
        /// <param name="scale">缩放系数（如 1/255）。</param>
        private static unsafe void PackSingleTileToNCHWFloat(IntPtr basePtr, int destOffsetFloats, Mat padded,
            int targetWidth, int targetHeight, int channels, bool swapRB, float scale)
        {
            if (channels != 3)
                throw new NotSupportedException($"仅支持 3 通道，当前: {channels}");
            if (padded.Type() != MatType.CV_8UC3)
                throw new NotSupportedException($"图类型须为 CV_8UC3，当前: {padded.Type()}");
            if (padded.Width != targetWidth || padded.Height != targetHeight)
                throw new ArgumentException($"图尺寸不匹配，期望 {targetWidth}x{targetHeight}，实际 {padded.Width}x{padded.Height}");

            byte* src = padded.DataPointer;
            int srcStride = (int)padded.Step();

            float* pBase = (float*)basePtr.ToPointer();
            int planeSize = targetHeight * targetWidth;
            float* dstB = pBase + destOffsetFloats + 0 * planeSize;
            float* dstG = pBase + destOffsetFloats + 1 * planeSize;
            float* dstR = pBase + destOffsetFloats + 2 * planeSize;

            if (swapRB)
            {
                for (int y = 0; y < targetHeight; y++)
                {
                    byte* row = src + y * srcStride;
                    float* dR = dstR + y * targetWidth;
                    float* dG = dstG + y * targetWidth;
                    float* dB = dstB + y * targetWidth;
                    for (int x = 0; x < targetWidth; x++)
                    {
                        int px = x * 3;
                        byte bB = row[px + 0], bG = row[px + 1], bR = row[px + 2];
                        *dR++ = bR * scale;
                        *dG++ = bG * scale;
                        *dB++ = bB * scale;
                    }
                }
            }
            else
            {
                for (int y = 0; y < targetHeight; y++)
                {
                    byte* row = src + y * srcStride;
                    float* dB = dstB + y * targetWidth;
                    float* dG = dstG + y * targetWidth;
                    float* dR = dstR + y * targetWidth;
                    for (int x = 0; x < targetWidth; x++)
                    {
                        int px = x * 3;
                        byte bB = row[px + 0], bG = row[px + 1], bR = row[px + 2];
                        *dB++ = bB * scale;
                        *dG++ = bG * scale;
                        *dR++ = bR * scale;
                    }
                }
            }
        }

        #region 内存优化清理方法
        /// <summary>
        /// 清理内存池和缓存，释放资源
        /// </summary>
        public void ClearMemoryOptimizations()
        {
            // 清理Mat对象池
            while (MatPool.TryDequeue(out var mat))
            {
                mat?.Dispose();
            }

            // 清理预处理参数缓存
            PreprocessParamsCache.Clear();

            // 张量内存池保留，因为可能还在使用中
            DebugLog("[MemoryOptimization] 内存优化清理完成 - Mat池和预处理参数缓存已清空");
        }

        /// <summary>
        /// 获取内存优化统计信息
        /// </summary>
        public string GetMemoryOptimizationStats()
        {
            var stats = $"Mat池大小: {MatPool.Count}, 预处理参数缓存大小: {PreprocessParamsCache.Count}, 张量内存池大小: {MemoryPool.Count}";
            DebugLog($"[MemoryStats] {stats}");
            return stats;
        }

        /// <summary>
        /// 获取详细的调试统计信息
        /// </summary>
        public string GetDetailedDebugStats()
        {
            var cacheKeys = PreprocessParamsCache.Keys.Take(5).ToList();
            var cacheInfo = string.Join(", ", cacheKeys.Select(k => $"{k.Item1}x{k.Item2}->{k.Item3}x{k.Item4}"));
            return $"内存优化统计:\n" +
                   $"- Mat对象池: {MatPool.Count} 个可复用对象\n" +
                   $"- 预处理缓存: {PreprocessParamsCache.Count} 个缓存项 [{cacheInfo}...]\n" +
                   $"- 张量内存池: {MemoryPool.Count} 个浮点数组\n" +
                   $"- 调试模式: {(DebugParsing ? "启用" : "禁用")}\n" +
                   $"- YAML缓存: {(YamlClass?.Length ?? 0)} 个类别名称";
        }
        #endregion
        #endregion
    }
}
