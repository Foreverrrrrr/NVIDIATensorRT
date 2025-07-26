using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NVIDIATensorRT.Internal
{
    /// <summary>
    /// 本机方法通过 P/Invoke 调用时是否发生异常的状态枚举
    /// </summary>
    public enum ExceptionStatus
    {
        /// <summary>
        /// 未发生异常
        /// </summary>
        NotOccurred = 0,

        /// <summary>
        /// 发生了一般异常
        /// </summary>
        Occurred = 1,

        /// <summary>
        /// 发生了 TensorRT 特定异常
        /// </summary>
        OccurredTRT = 2,

        /// <summary>
        /// 发生了 CUDA 特定异常
        /// </summary>
        OccurredCuda = 3
    }

}
