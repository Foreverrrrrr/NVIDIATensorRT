using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NVIDIATensorRT;
using NVIDIATensorRT.Custom;
using System.Text.RegularExpressions;
using System.Collections;
using System.Runtime.CompilerServices;

namespace NVIDIATensorRT.Deploy
{
    /// <summary>
    /// TensorRT推理引擎
    /// </summary>
    public unsafe class TensorRT
    {
        /// <summary>
        /// NVIDIA推理
        /// </summary>
        private readonly Nvinfer NVIDIAPredictor;

        /// <summary>
        /// 用于记录和处理与TensorRT相关的日志事件
        /// </summary>
        public event Action<string> NVIDIALogEvent;

        /// <summary>
        /// 张量内存池
        /// </summary>
        private readonly ConcurrentQueue<float[]> MemoryPool = new ConcurrentQueue<float[]>();

        /// <summary>
        /// 用于存储从YAML文件中加载的类别
        /// </summary>
        public string[] YamlClass { get; private set; }

        private float[] GetBuffer(int size) => MemoryPool.TryDequeue(out var buf) && buf.Length >= size ? buf : new float[size];

        private void ReturnBuffer(float[] buf) => MemoryPool.Enqueue(buf);

        public TensorRT() { }

        /// <summary>
        /// 加载 TensorRT 引擎文件（.engine）
        /// </summary>
        /// <param name="path_engine">TensorRT 引擎文件的路径</param>
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
        /// 加载 TensorRT 引擎并初始化输入输出张量结构
        /// </summary>
        /// <param name="path_engine">TensorRT 引擎文件路径</param>
        /// <param name="inputName">输入节点名称</param>
        /// <param name="outputName">输出节点名称</param>
        /// <param name="input">输入张量维度结构体</param>
        /// <param name="output">输出张量维度结构体</param>
        public TensorRT(string path_engine, string inputName, string outputName, out InputTensor input, out OutputTensor output)
        {
            if (!File.Exists(path_engine))
            {
                throw new FileNotFoundException($"指定的 TensorRT 引擎文件不存在: {path_engine}", path_engine);
            }
            NVIDIAPredictor = new Nvinfer(path_engine);
            Tensor(inputName, outputName, out input, out output);
        }

        /// <summary>
        /// 获取推理模型输入和输出张量的结构
        /// </summary>
        /// <param name="inputstring">模型输入节点名</param>
        /// <param name="outputstring">模型输出节点名</param>
        /// <param name="input">输入张量维度结构体</param>
        /// <param name="output">输出张量维度结构体</param>
        public void Tensor(string inputstring, string outputstring, out InputTensor input, out OutputTensor output)
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

        /// <summary>
        /// 预处理模型输入张量数据
        /// </summary>
        /// <param name="inputtensor">模型输入的张量结构</param>
        /// <param name="images">输入图像</param>
        /// <returns>数据指针及输入张量尺寸</returns>
        private unsafe SafeTensor ProcessBatch(ref InputTensor inputtensor, List<Mat> images)
        {
            if (images == null || images.Count < 1)
                throw new Exception("输入images错误");
            int actualBatch = images.Count;
            int singleImageSize = inputtensor.Channels * inputtensor.Height * inputtensor.Width;
            float[] batchData = GetBuffer(actualBatch * singleImageSize);
            try
            {
                int Width = inputtensor.Width; int Height = inputtensor.Height;
                Parallel.For(0, actualBatch, i =>
                {
                    using (Mat input_mat = CvDnn.BlobFromImage(
                        images[i],
                        1.0 / 255.0,
                        size: new OpenCvSharp.Size(Width, Height),
                        mean: new Scalar(0, 0, 0),
                        swapRB: true,
                        crop: false))
                    {
                        int offset = i * singleImageSize;
                        if (input_mat.Total() != singleImageSize)
                            throw new InvalidOperationException($"图像大小不匹配 {singleImageSize}, got {input_mat.Total()}");
                        fixed (float* pDest = &batchData[offset])
                        {
                            Buffer.MemoryCopy(
                                input_mat.Data.ToPointer(),
                                pDest,
                                singleImageSize * sizeof(float),
                                singleImageSize * sizeof(float));
                        }
                    }
                });
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
        /// 执行一次推理
        /// </summary>
        /// <param name="inputtensor">输入张量信息</param>
        /// <param name="mat">输入的图像数据</param>
        /// <returns>推理输出张量（float[]）</returns>
        public unsafe float[] InFerence(ref InputTensor inputtensor, Mat mat)
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
            float[] outputData = NVIDIAPredictor.GetInferenceResultArray("output0");
            intptr.Dispose();
            sw.Stop();
            long getResultTime = sw.ElapsedMilliseconds;
            totalWatch.Stop();
            long totalTime = totalWatch.ElapsedMilliseconds;
            NVIDIALogEvent?.Invoke($"输入张量处理耗时: {processBatchTime} 毫秒，加载推理数据耗时: {loadDataTime} 毫秒，推理耗时: {inferTime} 毫秒，获取张量结果耗时: {getResultTime} 毫秒，总计耗时: {totalTime} 毫秒");
            return outputData;
        }

        /// <summary>
        /// 执行一次推理
        /// </summary>
        /// <param name="inputtensor">输入张量信息</param>
        /// <param name="mat">输入的图像数据</param>
        /// <returns>返回输出张量非托管指针</returns>
        public unsafe InferenceTensor InFerencePtr(ref InputTensor inputtensor, Mat mat)
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
            int length = 0;
            InferenceTensor outputData = NVIDIAPredictor.GetInferenceResultPtr("output0", out length);
            intptr.Dispose();
            sw.Stop();
            long getResultTime = sw.ElapsedMilliseconds;
            totalWatch.Stop();
            long totalTime = totalWatch.ElapsedMilliseconds;
            NVIDIALogEvent?.Invoke($"输入张量处理耗时: {processBatchTime} 毫秒，加载推理数据耗时: {loadDataTime} 毫秒，推理耗时: {inferTime} 毫秒，获取张量结果耗时: {getResultTime} 毫秒，总计耗时: {totalTime} 毫秒");
            return outputData;
        }

        /// <summary>
        /// 执行一次推理（预热）
        /// </summary>
        /// <param name="inputtensor">输入张量信息</param>
        /// <param name="values">输入图像的列表，每个元素为一张图像</param>
        public void InFerence(ref InputTensor inputtensor, List<Mat> values)
        {
            if (NVIDIAPredictor == null)
                throw new Exception("NVIDIAPredictor初始化失败！");
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            var intptr = ProcessBatch(ref inputtensor, values);
            NVIDIAPredictor.LoadInferenceData("images", intptr.DataPtr, intptr.Length);
            NVIDIAPredictor.Infer();
            float[] outputData = NVIDIAPredictor.GetInferenceResultArray("output0");
            stopwatch.Stop();
            NVIDIALogEvent?.Invoke(stopwatch.ElapsedMilliseconds.ToString());
        }

        private int GetIndex(int batch, int channel, int detection, OutputTensor shape)
        {
            return batch * shape.Channels * shape.NumDetections
                 + channel * shape.NumDetections
                 + detection;
        }

        /// <summary>
        /// 解析模型的输出张量，返回经过置信度筛选与 NMS处理后的目标检测集合
        /// </summary>
        /// <param name="output">模型推理的输出张量</param>
        /// <param name="inshape">模型输入张量</param>
        /// <param name="outshape">模型输出张量</param>
        /// <param name="originalSize">原始输入图像尺寸</param>
        /// <param name="yamlpath">YAML配置文件</param>
        /// <param name="confidenceThreshold">目标框的置信度阈值</param>
        /// <param name="iouThreshold">NMS交并比阈值</param>
        /// <returns>处理后的检测结果列表，每个元素包含类别、置信度、边界框。</returns>
        /// <exception cref="ArgumentNullException">当 output==null 时抛出</exception>
        /// <exception cref="ArgumentException">当 batch size!=1 或通道数异常时抛出</exception>
        public List<DetectionResult> ParseYoloOutput(
        float[] output,
        InputTensor inshape,
        OutputTensor outshape,
        OpenCvSharp.Size originalSize,
        string yamlpath,
        float confidenceThreshold = 0.5f,
        float iouThreshold = 0.5f)
        {
            if (output == null) throw new ArgumentNullException(nameof(output), "输出数据不能为空！");
            if (outshape.BatchSize != 1) throw new ArgumentException($"预期的批次大小为 1，实际为 {outshape.BatchSize}！");
            int channels = outshape.Channels;
            int numDetections = outshape.NumDetections;
            if (channels < 5) throw new ArgumentException($"通道大小错误：{channels}！");
            int numClasses = channels - 4;
            var results = new List<DetectionResult>(numDetections);
            (float scale, float padX, float padY) = CalculatePreprocessParams(inshape, originalSize);
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
                if (maxClassScore < confidenceThreshold)
                    continue;
                RectF rect = ConvertToOriginalCoords(cx, cy, w, h, scale, padX, padY, originalSize);
                rect = ClampRect(rect, originalSize.Width, originalSize.Height);
                results.Add(new DetectionResult
                {
                    ClassId = bestClassId,
                    Confidence = maxClassScore,
                    BoundingBox = rect
                });
            }
            var classdescribe = GetClassNamesFromYaml(yamlpath);
            return ApplyClassAwareNMS(results, classdescribe, iouThreshold);
        }

        /// <summary>
        /// 解析模型的输出张量，返回经过置信度筛选与 NMS处理后的目标检测集合
        /// </summary>
        /// <param name="output">模型推理输出张量起始指针</param>
        /// <param name="inshape">输入张量的形状信息</param>
        /// <param name="outshape">输出张量的形状信息</param>
        /// <param name="originalSize">原始图像的尺寸</param>
        /// <param name="yamlpath">YAML配置文件</param>
        /// <param name="confidenceThreshold">目标框的置信度阈值</param>
        /// <param name="iouThreshold">NMS交并比阈值</param>
        /// <returns>检测结果的列表，每个结果包含类别 ID、置信度、边界框</returns>
        /// <exception cref="ArgumentNullException">当 output==null 时抛出</exception>
        /// <exception cref="ArgumentException">当 batch size!=1 或通道数异常时抛出</exception>
        public unsafe List<DetectionResult> ParseYoloOutput(
    float* output,
    InputTensor inshape,
    OutputTensor outshape,
    OpenCvSharp.Size originalSize,
    string yamlpath,
    float confidenceThreshold = 0.5f,
    float iouThreshold = 0.5f)
        {
            if (output == null) throw new ArgumentNullException(nameof(output), "输出数据不能为空！");
            if (outshape.BatchSize != 1) throw new ArgumentException($"预期的批次大小为 1，实际为 {outshape.BatchSize}！");
            int channels = outshape.Channels;
            int numDetections = outshape.NumDetections;
            if (channels < 5) throw new ArgumentException($"通道大小错误：{channels}！");
            int numClasses = channels - 4;
            var results = new List<DetectionResult>(numDetections);
            (float scale, float padX, float padY) = CalculatePreprocessParams(inshape, originalSize);
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
                if (maxClassScore < confidenceThreshold)
                    continue;
                RectF rect = ConvertToOriginalCoords(cx, cy, w, h, scale, padX, padY, originalSize);
                rect = ClampRect(rect, originalSize.Width, originalSize.Height);
                results.Add(new DetectionResult
                {
                    ClassId = bestClassId,
                    Confidence = maxClassScore,
                    BoundingBox = rect
                });
            }
            var classdescribe = GetClassNamesFromYaml(yamlpath);
            return ApplyClassAwareNMS(results, classdescribe, iouThreshold);
        }

        /// <summary>
        /// 计算预处理过程中的缩放和填充参数
        /// </summary>
        /// <param name="input">模型输入尺寸</param>
        /// <param name="originalSize">原始图像尺寸</param>
        /// <returns>
        /// scale：缩放比例（保持长宽比的最小缩放）
        /// padX：水平方向填充量（像素单位）
        /// padY：垂直方向填充量（像素单位）
        /// </returns>
        private static (float scale, float padX, float padY) CalculatePreprocessParams(InputTensor input, OpenCvSharp.Size originalSize)
        {
            float width = originalSize.Width;
            float height = originalSize.Height;
            float scaleW = input.Width / width;
            float scaleH = input.Height / height;
            float scale = (scaleW < scaleH) ? scaleW : scaleH;
            float padX = 0.5f * (input.Width - width * scale);
            float padY = 0.5f * (input.Height - height * scale);
            return (scale, padX, padY);
        }

        /// <summary>
        /// 将模型输出的归一化坐标转换为原始图像坐标系
        /// </summary>
        /// <param name="cx">中心点x坐标（相对于预处理后图像）</param>
        /// <param name="cy">中心点y坐标</param>
        /// <param name="w">边界框宽度</param>
        /// <param name="h">边界框高度</param>
        /// <param name="scale">预处理缩放比例</param>
        /// <param name="padX">预处理水平填充量</param>
        /// <param name="padY">预处理垂直填充量</param>
        /// <param name="originalSize">原始图像尺寸</param>
        /// <returns>原始坐标系下的RectF结构体</returns>
        private static RectF ConvertToOriginalCoords(
    float cx, float cy, float w, float h,
    float scale, float padX, float padY,
    OpenCvSharp.Size originalSize)
        {
            float halfW = 0.5f * w;
            float halfH = 0.5f * h;
            float invScale = 1.0f / scale;
            float x1 = (cx - halfW - padX) * invScale;
            float y1 = (cy - halfH - padY) * invScale;
            float x2 = (cx + halfW - padX) * invScale;
            float y2 = (cy + halfH - padY) * invScale;
            float maxW = originalSize.Width;
            float maxH = originalSize.Height;
            x1 = Clamp(x1, 0, maxW);
            y1 = Clamp(y1, 0, maxH);
            x2 = Clamp(x2, 0, maxW);
            y2 = Clamp(y2, 0, maxH);
            return new RectF(x1, y1, x2 - x1, y2 - y1);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Clamp(float val, float min, float max)
        {
            return (val < min) ? min : (val > max) ? max : val;
        }

        /// <summary>
        /// NMS非极大值抑制
        /// </summary>
        /// <param name="detections">原始检测结果列表</param>
        /// <param name="classdescribe">类别描述</param>
        /// <param name="iouThreshold">IoU重叠阈值</param>
        /// <returns>去重后的检测结果列表</returns>
        private List<DetectionResult> ApplyClassAwareNMS(
    List<DetectionResult> detections,
    string[] classdescribe,
    float iouThreshold)
        {
            if (detections == null || detections.Count == 0)
                return null;

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
            foreach (var kvp in classGroups)
            {
                var classId = kvp.Key;
                var classDetections = kvp.Value;
                classDetections.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
                int count = classDetections.Count;
                bool[] suppressed = new bool[count];
                for (int i = 0; i < count; i++)
                {
                    if (suppressed[i]) continue;
                    var current = classDetections[i];
                    if (classdescribe != null && current.ClassId < classdescribe.Length)
                        current.ClassDescribe = classdescribe[current.ClassId];
                    finalResults.Add(current);
                    RectF boxA = current.BoundingBox;
                    for (int j = i + 1; j < count; j++)
                    {
                        if (suppressed[j]) continue;
                        float iou = CalculateIoU(boxA, classDetections[j].BoundingBox);
                        if (iou > iouThreshold)
                        {
                            suppressed[j] = true;
                        }
                    }
                }
            }
            return finalResults;
        }


        /// <summary>
        /// 计算两个矩形框的交并比（Intersection over Union）
        /// </summary>
        /// <param name="a">第一个矩形框</param>
        /// <param name="b">第二个矩形框</param>
        /// <returns>IoU值（范围0-1）</returns>
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
        /// 解析 YAML 文件获取类名信息
        /// </summary>
        /// <param name="yamlFilePath">YAML 文件的路径，包含类名的配置</param>
        /// <returns>返回从 YAML 文件中提取的类名数组</returns>
        /// <exception cref="RegexMatchTimeoutException">如果正则表达式匹配失败异常</exception>
        public string[] GetClassNamesFromYaml(string yamlFilePath)
        {
            if (YamlClass == null)
            {
                if (yamlFilePath == null && !File.Exists(yamlFilePath))
                {
                    return null;
                }
                var yamlContent = File.ReadAllText(yamlFilePath);
                var match = Regex.Match(yamlContent, @"names:\s*\[([^\]]+)\]");
                if (match.Success)
                {
                    var namesString = match.Groups[1].Value;
                    YamlClass = namesString.Split(',')
                                                  .Select(name => name.Trim().Trim('\'', '"'))
                                                  .ToArray();
                    return YamlClass;
                }
                else
                    throw new RegexMatchTimeoutException(match.Name);
            }
            else
            {
                return YamlClass;
            }
        }
    }
}
