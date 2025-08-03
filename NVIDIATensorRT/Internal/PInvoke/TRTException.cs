using System;

namespace NVIDIATensorRT.Internal.PInvoke
{
    /// <summary>
    /// TensorRT 默认抛出的异常类
    /// </summary>
    [Serializable]
    // ReSharper disable once InconsistentNaming
    internal class TRTException : Exception
    {
        /// <summary>
        /// 错误状态的数值代码
        /// </summary>
        public ExceptionStatus status { get; set; }

        /// <summary>
        /// 错误描述信息
        /// </summary>
        public string err_msg { get; set; }

        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="status">错误状态的数值代码</param>
        /// <param name="err_msg">错误描述信息</param>
        public TRTException(ExceptionStatus status, string err_msg)
            : base(err_msg)
        {
            this.status = status;
            this.err_msg = err_msg;
        }
    }
}
