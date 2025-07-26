using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NVIDIATensorRT.Deploy
{
    /// <summary>
    /// 输入张量结构
    /// </summary>
    public struct InputTensor
    {
        /// <summary>
        /// 输入批次大小
        /// </summary>
        public int BatchSize { get; set; }
        /// <summary>
        /// 通道数
        /// </summary>
        public int Channels { get; set; }
        /// <summary>
        /// 高度
        /// </summary>
        public int Height { get; set; }
        /// <summary>
        /// 宽度
        /// </summary>
        public int Width { get; set; }
        /// <summary>
        /// 张量大小
        /// </summary>
        public int MagnitudeTensor => BatchSize * Channels * Height * Width;
    }
}
