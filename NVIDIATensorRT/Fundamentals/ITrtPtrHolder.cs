using System;

namespace NVIDIATensorRT
{
    /// <summary>
    /// 本机指针
    /// </summary>
    public interface ITrtPtrHolder
    {
        /// <summary>
        /// TensorRT数据指针
        /// </summary>
        IntPtr TrtPtr { get; }
    }
}
