using OpenCvSharp;
using System.Collections.Generic;

namespace NVIDIATensorRT.Deploy
{
    /// <summary>
    /// 输出张量结构
    /// </summary>
    public struct OutputTensor
    {
        /// <summary>
        /// 批次大小
        /// </summary>
        public int BatchSize { get; set; }

        /// <summary>
        /// 通道数
        /// <example></example>
        /// </summary>
        public int Channels { get; set; }

        /// <summary>
        /// 预测框数量
        /// <example></example>
        /// </summary>
        public int NumDetections { get; set; }

        /// <summary>
        /// 张量大小
        /// </summary>
        public int MagnitudeTensor => BatchSize * Channels * NumDetections;

        // ================= Callback 解析所需的可选元信息 =================
        /// <summary>
        /// 模型输入宽度（用于坐标还原的 scale/pad 计算）
        /// </summary>
        public int InputWidth { get; set; }

        /// <summary>
        /// 模型输入高度（用于坐标还原的 scale/pad 计算）
        /// </summary>
        public int InputHeight { get; set; }

        /// <summary>
        /// 每个 batch/tile 对应的原始 ROI 尺寸（与 InputWidth/Height 一起用于坐标还原）
        /// </summary>
        public Size[] InputImageSizeForRestore { get; set; }

        /// <summary>
        /// SAHI 分块的偏移与尺寸元信息；为空表示非 SAHI。
        /// </summary>
        public List<NVIDIATensorRT.Deploy.TensorRT.TileMeta> Tiles { get; set; }

        /// <summary>
        /// 解析所需的 YAML 路径（可选）
        /// </summary>
        public string YamlPath { get; set; }

        /// <summary>
        /// 置信度阈值（可选）
        /// </summary>
        public float ConfThreshold { get; set; }

        /// <summary>
        /// IoU 阈值（可选）
        /// </summary>
        public float IouThreshold { get; set; }
    }
}
