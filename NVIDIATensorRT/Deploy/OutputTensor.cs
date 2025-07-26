using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
    }
}